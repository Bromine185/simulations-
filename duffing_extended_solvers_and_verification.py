import contextlib
import dataclasses
import io
import json
import math
import pathlib
import sys
import time as wallclock
import tokenize

import numpy
import scipy.integrate
import scipy.linalg
import scipy.special
import sympy
import mpmath

verification_records = []


def record_verification(check_description, measured_value, threshold_value, passed):
    verification_records.append((check_description, measured_value, threshold_value, bool(passed)))
    return bool(passed)


time_symbol = sympy.Symbol('t', real=True)
damping_ratio_symbol = sympy.Symbol('eta', real=True)
natural_frequency_symbol = sympy.Symbol('omega_n', positive=True)
cubic_stiffness_symbol = sympy.Symbol('beta', positive=True)
forcing_amplitude_symbol = sympy.Symbol('F', real=True)
forcing_frequency_symbol = sympy.Symbol('Omega', positive=True)
initial_phase_symbol = sympy.Symbol('t_0', real=True)
position_symbol = sympy.Symbol('x', real=True)
velocity_symbol = sympy.Symbol('xdot', real=True)
position_function = sympy.Function('x', real=True)(time_symbol)

user_equation_terms = [
    sympy.diff(position_function, time_symbol, 2),
    -2 * damping_ratio_symbol * natural_frequency_symbol * sympy.diff(position_function, time_symbol),
    -natural_frequency_symbol**2 * position_function,
    cubic_stiffness_symbol * position_function**3,
]
forcing_expression = forcing_amplitude_symbol * sympy.cos(forcing_frequency_symbol * time_symbol)
governing_equation = sympy.Eq(sympy.Add(*user_equation_terms), forcing_expression)

acceleration_expression = sympy.expand(
    sympy.solve(governing_equation, sympy.diff(position_function, time_symbol, 2))[0]
    .subs(sympy.diff(position_function, time_symbol), velocity_symbol)
    .subs(position_function, position_symbol))
state_vector_field = sympy.Matrix([velocity_symbol, acceleration_expression])
state_jacobian = state_vector_field.jacobian([position_symbol, velocity_symbol])
phase_space_divergence = sympy.simplify(state_jacobian.trace())

energy_expression = (velocity_symbol**2 / 2 - natural_frequency_symbol**2 * position_symbol**2 / 2
                     + cubic_stiffness_symbol * position_symbol**4 / 4)
energy_rate_along_flow = (sympy.diff(energy_expression, position_symbol) * velocity_symbol
                          + sympy.diff(energy_expression, velocity_symbol) * acceleration_expression)
claimed_power_balance = (2 * damping_ratio_symbol * natural_frequency_symbol * velocity_symbol**2
                         + forcing_expression * velocity_symbol)

unforced_acceleration = acceleration_expression.subs(forcing_amplitude_symbol, 0)
equilibrium_positions = sympy.solve(unforced_acceleration.subs(velocity_symbol, 0), position_symbol)
equilibrium_eigenvalues = {
    equilibrium: [sympy.factor(sympy.simplify(eigenvalue)) for eigenvalue in
                  state_jacobian.subs({position_symbol: equilibrium, velocity_symbol: 0}).eigenvals()]
    for equilibrium in equilibrium_positions
}

sine_amplitude, cosine_amplitude, delta_amplitude = sympy.symbols('s c d', real=True)
elliptic_parameter = sympy.Symbol('m', positive=True)
turning_amplitude = sympy.Symbol('A', positive=True)
time_scale = sympy.Symbol('lambda', positive=True)
pythagorean_ideal = [cosine_amplitude**2 + sine_amplitude**2 - 1,
                     delta_amplitude**2 + elliptic_parameter * sine_amplitude**2 - 1]


def jacobi_derivative(expression):
    return (sympy.diff(expression, sine_amplitude) * cosine_amplitude * delta_amplitude
            - sympy.diff(expression, cosine_amplitude) * sine_amplitude * delta_amplitude
            - sympy.diff(expression, delta_amplitude) * elliptic_parameter * sine_amplitude * cosine_amplitude)


def conservative_residual_is_zero(candidate_solution, time_scale_value, elliptic_parameter_value):
    residual = (time_scale**2 * jacobi_derivative(jacobi_derivative(candidate_solution))
                - natural_frequency_symbol**2 * candidate_solution
                + cubic_stiffness_symbol * candidate_solution**3)
    remainder = sympy.reduced(sympy.expand(residual), pythagorean_ideal,
                              cosine_amplitude, delta_amplitude, sine_amplitude)[1]
    return sympy.simplify(remainder.subs({time_scale: time_scale_value,
                                          elliptic_parameter: elliptic_parameter_value})) == 0


intrawell_time_scale = turning_amplitude * sympy.sqrt(cubic_stiffness_symbol / 2)
intrawell_parameter = 2 - 2 * natural_frequency_symbol**2 / (cubic_stiffness_symbol * turning_amplitude**2)
interwell_time_scale = sympy.sqrt(cubic_stiffness_symbol * turning_amplitude**2 - natural_frequency_symbol**2)
interwell_parameter = cubic_stiffness_symbol * turning_amplitude**2 / (
    2 * (cubic_stiffness_symbol * turning_amplitude**2 - natural_frequency_symbol**2))

homoclinic_position = (sympy.sqrt(2 / cubic_stiffness_symbol) * natural_frequency_symbol
                       * sympy.sech(natural_frequency_symbol * time_symbol))
homoclinic_velocity = sympy.diff(homoclinic_position, time_symbol)

hyperbolic_tangent_variable = sympy.Symbol('w', real=True)
damping_melnikov_integral = sympy.integrate(
    2 * natural_frequency_symbol**3 / cubic_stiffness_symbol * hyperbolic_tangent_variable**2,
    (hyperbolic_tangent_variable, -1, 1))
forcing_melnikov_integral = (-sympy.sqrt(2 / cubic_stiffness_symbol) * sympy.pi * forcing_frequency_symbol
                             * sympy.sech(sympy.pi * forcing_frequency_symbol / (2 * natural_frequency_symbol)))
melnikov_function = sympy.simplify(
    2 * damping_ratio_symbol * natural_frequency_symbol * damping_melnikov_integral
    - forcing_amplitude_symbol * forcing_melnikov_integral * sympy.sin(forcing_frequency_symbol * initial_phase_symbol))
melnikov_threshold_expression = sympy.simplify(
    sympy.Abs(2 * damping_ratio_symbol * natural_frequency_symbol * damping_melnikov_integral)
    / sympy.Abs(forcing_melnikov_integral))
melnikov_threshold_function = sympy.lambdify(
    (damping_ratio_symbol, natural_frequency_symbol, cubic_stiffness_symbol, forcing_frequency_symbol),
    melnikov_threshold_expression, 'numpy')


def verify_symbolic_derivations():
    energy_identity_residual = sympy.simplify(energy_rate_along_flow - claimed_power_balance)
    record_verification('Symbolic: dH/dt along the flow equals 2*eta*omega_n*xdot^2 + F*cos(Omega t)*xdot',
                        str(energy_identity_residual), '0', energy_identity_residual == 0)
    divergence_residual = sympy.simplify(phase_space_divergence - 2 * damping_ratio_symbol * natural_frequency_symbol)
    record_verification('Symbolic: phase-space divergence equals 2*eta*omega_n',
                        str(phase_space_divergence), '2*eta*omega_n', divergence_residual == 0)
    intrawell_solution_proved = conservative_residual_is_zero(
        turning_amplitude * delta_amplitude, intrawell_time_scale, intrawell_parameter)
    record_verification('Symbolic: A*dn(lambda t | m) solves the conservative equation (Groebner reduction)',
                        str(intrawell_solution_proved), 'True', intrawell_solution_proved)
    interwell_solution_proved = conservative_residual_is_zero(
        turning_amplitude * cosine_amplitude, interwell_time_scale, interwell_parameter)
    record_verification('Symbolic: A*cn(lambda t | m) solves the conservative equation (Groebner reduction)',
                        str(interwell_solution_proved), 'True', interwell_solution_proved)
    wrong_time_scale_rejected = not conservative_residual_is_zero(
        turning_amplitude * delta_amplitude, intrawell_time_scale * sympy.Rational(101, 100), intrawell_parameter)
    record_verification('Symbolic: Groebner proof rejects a deliberately wrong dn time scale (negative control)',
                        str(wrong_time_scale_rejected), 'True', wrong_time_scale_rejected)
    homoclinic_solution_proved = sympy.simplify(
        (sympy.diff(homoclinic_position, time_symbol, 2) - natural_frequency_symbol**2 * homoclinic_position
         + cubic_stiffness_symbol * homoclinic_position**3).rewrite(sympy.exp)) == 0
    record_verification('Symbolic: sqrt(2/beta)*omega_n*sech(omega_n t) is the homoclinic orbit',
                        str(homoclinic_solution_proved), 'True', homoclinic_solution_proved)
    probe_values = {natural_frequency_symbol: sympy.Rational(13, 10), cubic_stiffness_symbol: sympy.Rational(7, 10),
                    forcing_frequency_symbol: sympy.Rational(11, 10)}
    with mpmath.workdps(45):
        homoclinic_velocity_numeric = sympy.lambdify(time_symbol, homoclinic_velocity.subs(probe_values), 'mpmath')
        probe_forcing_frequency = mpmath.mpf(11) / 10
        quadrature_damping = 2 * mpmath.quad(lambda time_value: homoclinic_velocity_numeric(time_value)**2,
                                             [0, 5, 20, mpmath.inf])
        quadrature_forcing = 2 * mpmath.quadosc(
            lambda time_value: homoclinic_velocity_numeric(time_value) * mpmath.sin(probe_forcing_frequency * time_value),
            [0, mpmath.inf], omega=probe_forcing_frequency)
        closed_form_damping = mpmath.mpf(str(sympy.N(damping_melnikov_integral.subs(probe_values), 45)))
        closed_form_forcing = mpmath.mpf(str(sympy.N(forcing_melnikov_integral.subs(probe_values), 45)))
        melnikov_quadrature_discrepancy = float(max(abs(quadrature_damping - closed_form_damping),
                                                    abs(quadrature_forcing - closed_form_forcing)))
    record_verification('Melnikov integrals: closed form vs 45-digit quadrature',
                        melnikov_quadrature_discrepancy, 1e-30, melnikov_quadrature_discrepancy < 1e-30)
    negative_damping_ratio = sympy.Symbol('eta_dissipative', negative=True)
    tangency_value = sympy.simplify(melnikov_function.subs(
        forcing_amplitude_symbol, melnikov_threshold_expression).subs(
        sympy.sin(forcing_frequency_symbol * initial_phase_symbol), 1).subs(damping_ratio_symbol, negative_damping_ratio))
    record_verification('Symbolic: max over t_0 of M(t_0) vanishes exactly at F = F_c (homoclinic tangency)',
                        str(tangency_value), '0', tangency_value == 0)


def classification_from_eigenvalues(eigenvalues):
    eigenvalues = numpy.asarray(eigenvalues, dtype=complex)
    real_parts, imaginary_parts = eigenvalues.real, eigenvalues.imag
    if numpy.all(numpy.abs(real_parts) < 1e-12):
        return 'center'
    if numpy.all(numpy.abs(imaginary_parts) < 1e-12):
        if real_parts.min() < 0 < real_parts.max():
            return 'saddle'
        return 'stable node' if real_parts.max() < 0 else 'unstable node'
    return 'stable focus' if real_parts.max() < 0 else 'unstable focus'


@dataclasses.dataclass(frozen=True)
class OscillatorParameters:
    damping_ratio: float = 0.0
    natural_frequency: float = 1.0
    cubic_stiffness: float = 1.0
    forcing_amplitude: float = 0.0
    forcing_frequency: float = 1.0

    @property
    def damping_coefficient(self):
        return 2 * self.damping_ratio * self.natural_frequency

    @property
    def well_position(self):
        return self.natural_frequency / numpy.sqrt(self.cubic_stiffness)

    def acceleration(self, time, position, velocity):
        return (self.damping_coefficient * velocity + self.natural_frequency**2 * position
                - self.cubic_stiffness * position**3
                + self.forcing_amplitude * numpy.cos(self.forcing_frequency * time))

    def vector_field(self, time, state):
        return numpy.array([state[1], self.acceleration(time, state[0], state[1])])

    def jacobian(self, time, state):
        return numpy.array([[0.0, 1.0],
                            [self.natural_frequency**2 - 3 * self.cubic_stiffness * state[0]**2, self.damping_coefficient]])

    def energy(self, position, velocity):
        return (0.5 * velocity**2 - 0.5 * self.natural_frequency**2 * position**2
                + 0.25 * self.cubic_stiffness * position**4)

    def input_power(self, time, position, velocity):
        return (self.damping_coefficient * velocity**2
                + self.forcing_amplitude * numpy.cos(self.forcing_frequency * time) * velocity)

    def equilibria(self):
        summary = []
        for position in (-self.well_position, 0.0, self.well_position):
            eigenvalues = numpy.linalg.eigvals(self.jacobian(0.0, numpy.array([position, 0.0])))
            summary.append({'position': position, 'eigenvalues': eigenvalues,
                            'classification': classification_from_eigenvalues(eigenvalues)})
        return summary

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)


def exact_conservative_solution(parameters, turning_amplitude, time_values, decimal_digits=None):
    if decimal_digits is None:
        natural_frequency, cubic_stiffness, amplitude = (float(parameters.natural_frequency),
                                                         float(parameters.cubic_stiffness), float(turning_amplitude))
        time_values = numpy.asarray(time_values, dtype=float)
        separatrix_amplitude_squared = 2 * natural_frequency**2 / cubic_stiffness
        if abs(amplitude**2 - separatrix_amplitude_squared) < 1e-15 * separatrix_amplitude_squared:
            argument = natural_frequency * time_values
            return (amplitude / numpy.cosh(argument), -amplitude * natural_frequency * numpy.tanh(argument) / numpy.cosh(argument),
                    numpy.inf)
        if amplitude**2 > separatrix_amplitude_squared:
            time_scale_value = math.sqrt(cubic_stiffness * amplitude**2 - natural_frequency**2)
            parameter_value = cubic_stiffness * amplitude**2 / (2 * time_scale_value**2)
            elliptic_sine, elliptic_cosine, elliptic_delta, _ = scipy.special.ellipj(time_scale_value * time_values, parameter_value)
            return (amplitude * elliptic_cosine, -amplitude * time_scale_value * elliptic_sine * elliptic_delta,
                    4 * scipy.special.ellipk(parameter_value) / time_scale_value)
        time_scale_value = amplitude * math.sqrt(cubic_stiffness / 2)
        parameter_value = 2 - 2 * natural_frequency**2 / (cubic_stiffness * amplitude**2)
        elliptic_sine, elliptic_cosine, elliptic_delta, _ = scipy.special.ellipj(time_scale_value * time_values, parameter_value)
        return (amplitude * elliptic_delta, -amplitude * time_scale_value * parameter_value * elliptic_sine * elliptic_cosine,
                2 * scipy.special.ellipk(parameter_value) / time_scale_value)
    with mpmath.workdps(max(decimal_digits, mpmath.mp.dps)):
        natural_frequency = mpmath.mpf(parameters.natural_frequency)
        cubic_stiffness = mpmath.mpf(parameters.cubic_stiffness)
        amplitude = mpmath.mpf(turning_amplitude)
        time_values = [mpmath.mpf(time_value) for time_value in time_values]
        if amplitude**2 > 2 * natural_frequency**2 / cubic_stiffness:
            time_scale_value = mpmath.sqrt(cubic_stiffness * amplitude**2 - natural_frequency**2)
            parameter_value = cubic_stiffness * amplitude**2 / (2 * time_scale_value**2)
            positions = [amplitude * mpmath.ellipfun('cn', time_scale_value * time_value, m=parameter_value) for time_value in time_values]
            velocities = [-amplitude * time_scale_value * mpmath.ellipfun('sn', time_scale_value * time_value, m=parameter_value)
                          * mpmath.ellipfun('dn', time_scale_value * time_value, m=parameter_value) for time_value in time_values]
            return positions, velocities, 4 * mpmath.ellipk(parameter_value) / time_scale_value
        time_scale_value = amplitude * mpmath.sqrt(cubic_stiffness / 2)
        parameter_value = 2 - 2 * natural_frequency**2 / (cubic_stiffness * amplitude**2)
        positions = [amplitude * mpmath.ellipfun('dn', time_scale_value * time_value, m=parameter_value) for time_value in time_values]
        velocities = [-amplitude * time_scale_value * parameter_value * mpmath.ellipfun('sn', time_scale_value * time_value, m=parameter_value)
                      * mpmath.ellipfun('cn', time_scale_value * time_value, m=parameter_value) for time_value in time_values]
        return positions, velocities, 2 * mpmath.ellipk(parameter_value) / time_scale_value


def verify_model_consistency():
    symbolic_acceleration_function = sympy.lambdify(
        (time_symbol, position_symbol, velocity_symbol, damping_ratio_symbol, natural_frequency_symbol,
         cubic_stiffness_symbol, forcing_amplitude_symbol, forcing_frequency_symbol), acceleration_expression, 'numpy')
    consistency_samples = numpy.random.default_rng(20261006).uniform(-2, 2, size=(8, 2000))
    consistency_samples[[4, 5, 7]] = numpy.abs(consistency_samples[[4, 5, 7]]) + 0.1
    consistency_parameters = OscillatorParameters(*consistency_samples[3:8])
    handwritten_acceleration = consistency_parameters.acceleration(*consistency_samples[0:3])
    symbolic_acceleration = symbolic_acceleration_function(*consistency_samples)
    lambdify_discrepancy = float(numpy.max(numpy.abs(handwritten_acceleration - symbolic_acceleration)
                                           / (1 + numpy.abs(symbolic_acceleration))))
    record_verification('Hand-written right-hand side vs SymPy-generated right-hand side (2000 random states)',
                        lambdify_discrepancy, 1e-14, lambdify_discrepancy < 1e-14)

    eigenvalue_discrepancies = []
    for damping_ratio_value in (-0.3, -0.05, 0.0, 0.05, 0.3, 2.5):
        numeric_parameters = OscillatorParameters(damping_ratio=damping_ratio_value, natural_frequency=1.3, cubic_stiffness=0.7)
        substitutions = {damping_ratio_symbol: damping_ratio_value, natural_frequency_symbol: 1.3, cubic_stiffness_symbol: 0.7}
        for equilibrium, symbolic_eigenvalues in equilibrium_eigenvalues.items():
            numeric_eigenvalues = numpy.sort_complex(numpy.linalg.eigvals(numeric_parameters.jacobian(
                0.0, numpy.array([float(equilibrium.subs(substitutions)), 0.0]))))
            formula_eigenvalues = numpy.sort_complex(numpy.array(
                [complex(sympy.N(value.subs(substitutions))) for value in symbolic_eigenvalues]))
            eigenvalue_discrepancies.append(numpy.max(numpy.abs(numeric_eigenvalues - formula_eigenvalues)))
    eigenvalue_discrepancy = float(max(eigenvalue_discrepancies))
    record_verification('Numerical Jacobian eigenvalues vs symbolic eigenvalue formulas at all three equilibria',
                        eigenvalue_discrepancy, 1e-12, eigenvalue_discrepancy < 1e-12)

    expected_classifications = {
        -2.5: ['stable node', 'saddle', 'stable node'],
        -0.3: ['stable focus', 'saddle', 'stable focus'],
        0.0: ['center', 'saddle', 'center'],
        0.3: ['unstable focus', 'saddle', 'unstable focus'],
        2.5: ['unstable node', 'saddle', 'unstable node'],
    }
    classification_mismatches = sum(
        [equilibrium['classification'] for equilibrium in
         OscillatorParameters(damping_ratio=damping_ratio_value).equilibria()] != expected_classifications[damping_ratio_value]
        for damping_ratio_value in expected_classifications)
    record_verification('Equilibrium classification (node / focus / center / saddle) across the sign and size of eta',
                        classification_mismatches, 0, classification_mismatches == 0)

    with mpmath.workdps(60):
        residual_probe_parameters = OscillatorParameters(natural_frequency=1.0, cubic_stiffness=1.0)
        exact_solution_residuals = []
        for probe_amplitude in ('1.2', '1.7'):
            for probe_time in ('0.37', '2.9'):
                def probe_position(time_value):
                    return exact_conservative_solution(residual_probe_parameters, mpmath.mpf(probe_amplitude), [time_value], 60)[0][0]
                probe_time_value = mpmath.mpf(probe_time)
                position_value = probe_position(probe_time_value)
                exact_solution_residuals.append(abs(mpmath.diff(probe_position, probe_time_value, 2)
                                                    - position_value + position_value**3))
        exact_solution_residual = float(max(exact_solution_residuals))
    record_verification('Jacobi elliptic exact solutions satisfy the ODE at 60 digits (mpmath differentiation)',
                        exact_solution_residual, 1e-45, exact_solution_residual < 1e-45)


@dataclasses.dataclass
class OscillatorSolution:
    method: str
    time: numpy.ndarray
    position: numpy.ndarray
    velocity: numpy.ndarray
    accepted_steps: int = 0
    rejected_steps: int = 0
    function_evaluations: int = 0
    wall_time: float = 0.0
    dense_output: object = None
    series_segments: object = None
    states: object = None

    def __call__(self, query_times):
        if self.dense_output is None:
            raise ValueError(f'{self.method} was run without dense output')
        return self.dense_output(query_times)


class TaylorSeriesIntegrator:
    def __init__(self, parameters, tolerance=1e-16, series_order=None, decimal_digits=None):
        self.parameters = parameters
        self.tolerance = tolerance
        self.decimal_digits = decimal_digits
        self.series_order = series_order or max(6, int(math.ceil(-math.log(tolerance) / 2)) + 2)
        self.coefficient_type = object if decimal_digits else float
        self.convert = mpmath.mpf if decimal_digits else float
        self.cosine = mpmath.cos if decimal_digits else math.cos
        self.sine = mpmath.sin if decimal_digits else math.sin
        self.safety_factor = math.exp(-0.7 / (self.series_order - 1))

    def precision_context(self):
        return mpmath.workdps(self.decimal_digits) if self.decimal_digits else contextlib.nullcontext()

    def series_coefficients(self, time, position, velocity):
        order = self.series_order
        damping_coefficient = 2 * self.convert(self.parameters.damping_ratio) * self.convert(self.parameters.natural_frequency)
        linear_stiffness = self.convert(self.parameters.natural_frequency)**2
        cubic_stiffness = self.convert(self.parameters.cubic_stiffness)
        forcing_amplitude = self.convert(self.parameters.forcing_amplitude)
        forcing_frequency = self.convert(self.parameters.forcing_frequency)
        position_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        velocity_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        square_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        cube_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        cosine_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        sine_coefficients = numpy.zeros(order + 1, dtype=self.coefficient_type)
        position_coefficients[0], velocity_coefficients[0] = position, velocity
        cosine_coefficients[0] = self.cosine(forcing_frequency * time)
        sine_coefficients[0] = self.sine(forcing_frequency * time)
        for index in range(order):
            reversed_position = position_coefficients[index::-1]
            square_coefficients[index] = numpy.dot(position_coefficients[:index + 1], reversed_position)
            cube_coefficients[index] = numpy.dot(square_coefficients[:index + 1], reversed_position)
            position_coefficients[index + 1] = velocity_coefficients[index] / (index + 1)
            velocity_coefficients[index + 1] = (damping_coefficient * velocity_coefficients[index]
                                                + linear_stiffness * position_coefficients[index]
                                                - cubic_stiffness * cube_coefficients[index]
                                                + forcing_amplitude * cosine_coefficients[index]) / (index + 1)
            cosine_coefficients[index + 1] = -forcing_frequency * sine_coefficients[index] / (index + 1)
            sine_coefficients[index + 1] = forcing_frequency * cosine_coefficients[index] / (index + 1)
        return position_coefficients, velocity_coefficients

    def adaptive_step_size(self, position_coefficients, velocity_coefficients, remaining_interval):
        scale = max(1, abs(position_coefficients[0]), abs(velocity_coefficients[0]))
        candidate_steps = []
        for index in (self.series_order - 1, self.series_order):
            magnitude = max(abs(position_coefficients[index]), abs(velocity_coefficients[index]))
            if magnitude > 0:
                candidate_steps.append(float((self.tolerance * scale / magnitude) ** (1.0 / index)))
        if not candidate_steps:
            return remaining_interval
        return min(self.safety_factor * min(candidate_steps), remaining_interval)

    @staticmethod
    def evaluate_series(coefficient_rows, offsets):
        result = coefficient_rows[..., -1]
        for column in range(coefficient_rows.shape[-1] - 2, -1, -1):
            result = result * offsets + coefficient_rows[..., column]
        return result

    def integrate(self, initial_position, initial_velocity, start_time, end_time, number_of_steps=None):
        with self.precision_context():
            started = wallclock.perf_counter()
            time, end_time = self.convert(start_time), self.convert(end_time)
            position, velocity = self.convert(initial_position), self.convert(initial_velocity)
            direction = 1 if end_time >= time else -1
            fixed_step = abs(end_time - time) / number_of_steps if number_of_steps else None
            step_start_times, step_sizes, position_rows, velocity_rows = [], [], [], []
            node_times, node_positions, node_velocities = [time], [position], [velocity]
            while direction * (end_time - time) > 0:
                position_coefficients, velocity_coefficients = self.series_coefficients(time, position, velocity)
                remaining_interval = abs(end_time - time)
                step = min(fixed_step, remaining_interval) if fixed_step else self.adaptive_step_size(
                    position_coefficients, velocity_coefficients, remaining_interval)
                if fixed_step and len(step_sizes) == number_of_steps - 1:
                    step = remaining_interval
                signed_step = direction * step
                position = self.evaluate_series(position_coefficients, signed_step)
                velocity = self.evaluate_series(velocity_coefficients, signed_step)
                step_start_times.append(time)
                step_sizes.append(signed_step)
                position_rows.append(position_coefficients)
                velocity_rows.append(velocity_coefficients)
                time = end_time if step == remaining_interval else time + signed_step
                node_times.append(time)
                node_positions.append(position)
                node_velocities.append(velocity)
            elapsed = wallclock.perf_counter() - started
        position_table = numpy.array(position_rows, dtype=self.coefficient_type)
        velocity_table = numpy.array(velocity_rows, dtype=self.coefficient_type)
        start_table = numpy.array(step_start_times, dtype=self.coefficient_type)
        float_starts = numpy.array([float(value) for value in step_start_times])

        def dense_output(query_times):
            with self.precision_context():
                query_array = numpy.array([self.convert(value) for value in numpy.atleast_1d(query_times)],
                                          dtype=self.coefficient_type)
                float_queries = numpy.array([float(value) for value in query_array])
                if direction > 0:
                    step_indices = numpy.clip(numpy.searchsorted(float_starts, float_queries, side='right') - 1, 0, len(float_starts) - 1)
                else:
                    step_indices = numpy.clip(numpy.searchsorted(-float_starts, -float_queries, side='right') - 1, 0, len(float_starts) - 1)
                offsets = query_array - start_table[step_indices]
                return (self.evaluate_series(position_table[step_indices], offsets),
                        self.evaluate_series(velocity_table[step_indices], offsets))

        return OscillatorSolution(
            method='Taylor series (own)', time=numpy.array(node_times, dtype=self.coefficient_type),
            position=numpy.array(node_positions, dtype=self.coefficient_type),
            velocity=numpy.array(node_velocities, dtype=self.coefficient_type),
            accepted_steps=len(step_sizes), function_evaluations=len(step_sizes) * self.series_order,
            wall_time=elapsed, dense_output=dense_output,
            series_segments=(start_table, numpy.array(step_sizes, dtype=self.coefficient_type), position_table, velocity_table))


@dataclasses.dataclass(frozen=True)
class ButcherTableau:
    name: str
    order: int
    stage_matrix: numpy.ndarray
    weights: numpy.ndarray
    nodes: numpy.ndarray
    embedded_weights: numpy.ndarray = None
    embedded_order: int = None

    @property
    def first_same_as_last(self):
        return bool(numpy.allclose(self.stage_matrix[-1], self.weights) and self.nodes[-1] == 1)


classical_runge_kutta_tableau = ButcherTableau(
    name='Classical Runge-Kutta 4 (own)', order=4,
    stage_matrix=numpy.array([[0, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 1, 0]], dtype=float),
    weights=numpy.array([1, 2, 2, 1], dtype=float) / 6, nodes=numpy.array([0, 0.5, 0.5, 1]))
dormand_prince_tableau = ButcherTableau(
    name='Dormand-Prince 5(4) (own)', order=5, embedded_order=4,
    stage_matrix=numpy.array([
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]]),
    weights=numpy.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]),
    embedded_weights=numpy.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]),
    nodes=numpy.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1]))


def gauss_legendre_tableau(number_of_stages):
    legendre_nodes, legendre_weights = numpy.polynomial.legendre.leggauss(number_of_stages)
    nodes, weights = (legendre_nodes + 1) / 2, legendre_weights / 2
    stage_matrix = numpy.empty((number_of_stages, number_of_stages))
    for column in range(number_of_stages):
        other_nodes = numpy.delete(nodes, column)
        lagrange_basis = (numpy.polynomial.Polynomial.fromroots(other_nodes) if other_nodes.size
                          else numpy.polynomial.Polynomial([1.0]))
        antiderivative = (lagrange_basis / lagrange_basis(nodes[column])).integ()
        stage_matrix[:, column] = antiderivative(nodes) - antiderivative(0.0)
    return ButcherTableau(name=f'Gauss-Legendre {number_of_stages}-stage (own)', order=2 * number_of_stages,
                          stage_matrix=stage_matrix, weights=weights, nodes=nodes)


def scaled_error_norm(error, state, new_state, relative_tolerance, absolute_tolerance):
    scale = absolute_tolerance + relative_tolerance * numpy.maximum(numpy.abs(state), numpy.abs(new_state))
    return float(numpy.sqrt(numpy.mean((error / scale)**2)))


def initial_step_size(vector_field, start_time, state, derivative, order, relative_tolerance, absolute_tolerance, direction):
    scale = absolute_tolerance + relative_tolerance * numpy.abs(state)
    state_norm = numpy.sqrt(numpy.mean((state / scale)**2))
    derivative_norm = numpy.sqrt(numpy.mean((derivative / scale)**2))
    trial_step = 1e-6 if min(state_norm, derivative_norm) < 1e-5 else 0.01 * state_norm / derivative_norm
    trial_derivative = vector_field(start_time + direction * trial_step, state + direction * trial_step * derivative)
    second_derivative_norm = numpy.sqrt(numpy.mean(((trial_derivative - derivative) / scale)**2)) / trial_step
    largest_norm = max(derivative_norm, second_derivative_norm)
    proposal = (max(1e-6, trial_step * 1e-3) if largest_norm <= 1e-15
                else (0.01 / largest_norm) ** (1.0 / (order + 1)))
    return min(100 * trial_step, proposal)


def integrate_explicit_runge_kutta(vector_field, initial_state, start_time, end_time, tableau, number_of_steps=None,
                                   relative_tolerance=1e-8, absolute_tolerance=1e-10, output_times=None,
                                   record_every_step=True, maximum_steps=50_000_000):
    started = wallclock.perf_counter()
    state = numpy.array(initial_state, dtype=float)
    time = float(start_time)
    direction = 1.0 if end_time >= start_time else -1.0
    requested_outputs = set() if output_times is None else set(numpy.asarray(output_times, dtype=float).tolist())
    stop_times = sorted((stop for stop in requested_outputs | {float(end_time)}
                         if direction * (stop - time) > 0 and direction * (end_time - stop) >= 0),
                        key=lambda stop: direction * stop)
    record_initial = record_every_step or output_times is None or time in requested_outputs
    adaptive = number_of_steps is None
    stage_count = len(tableau.nodes)
    stages = numpy.empty((stage_count, state.size))
    derivative = vector_field(time, state)
    evaluations = 1
    if adaptive:
        error_order = min(tableau.order, tableau.embedded_order)
        controller_beta = 0.04
        controller_alpha = 1.0 / (error_order + 1) - 0.75 * controller_beta
        proposed_step = initial_step_size(vector_field, time, state, derivative, error_order,
                                          relative_tolerance, absolute_tolerance, direction)
        evaluations += 1
        previous_error = 1e-4
    else:
        proposed_step = abs(end_time - start_time) / number_of_steps
    recorded_times = [time] if record_initial else []
    recorded_states = [state.copy()] if record_initial else []
    accepted, rejected = 0, 0
    next_stop_index = 0
    while next_stop_index < len(stop_times) and accepted + rejected < maximum_steps:
        target = stop_times[next_stop_index]
        remaining = direction * (target - time)
        hits_stop = proposed_step >= remaining * (1 - 1e-12)
        step = remaining if hits_stop else proposed_step
        signed_step = direction * step
        stages[0] = derivative
        for stage_index in range(1, stage_count):
            stage_state = state + signed_step * (tableau.stage_matrix[stage_index, :stage_index] @ stages[:stage_index])
            stages[stage_index] = vector_field(time + tableau.nodes[stage_index] * signed_step, stage_state)
        evaluations += stage_count - 1
        new_state = state + signed_step * (tableau.weights @ stages)
        if adaptive:
            error_estimate = signed_step * ((tableau.weights - tableau.embedded_weights) @ stages)
            error_norm = scaled_error_norm(error_estimate, state, new_state, relative_tolerance, absolute_tolerance)
            if error_norm > 1.0:
                rejected += 1
                proposed_step = step * max(0.2, 0.9 * error_norm ** (-1.0 / (error_order + 1)))
                continue
            growth = 10.0 if error_norm == 0 else min(10.0, max(0.2, 0.9 * error_norm ** (-controller_alpha)
                                                                 * previous_error ** controller_beta))
            if not hits_stop or step >= proposed_step * 0.5:
                proposed_step = step * growth
            previous_error = max(error_norm, 1e-4)
        accepted += 1
        time = target if hits_stop else time + signed_step
        state = new_state
        if tableau.first_same_as_last:
            derivative = stages[-1].copy()
        else:
            derivative = vector_field(time, state)
            evaluations += 1
        if record_every_step or (hits_stop and target in requested_outputs):
            recorded_times.append(time)
            recorded_states.append(state.copy())
        if hits_stop:
            next_stop_index += 1
    recorded_states = numpy.array(recorded_states)
    return OscillatorSolution(method=tableau.name, time=numpy.array(recorded_times), position=recorded_states[:, 0],
                              velocity=recorded_states[:, 1], accepted_steps=accepted, rejected_steps=rejected,
                              function_evaluations=evaluations, wall_time=wallclock.perf_counter() - started,
                              states=recorded_states)


def integrate_gauss_legendre(vector_field, jacobian, initial_state, start_time, end_time, number_of_stages,
                             number_of_steps, record_stride=1, maximum_newton_iterations=30):
    started = wallclock.perf_counter()
    tableau = gauss_legendre_tableau(number_of_stages)
    state = numpy.array(initial_state, dtype=float)
    dimension = state.size
    step = (end_time - start_time) / number_of_steps
    identity = numpy.eye(number_of_stages * dimension)
    recorded_times, recorded_states = [float(start_time)], [state.copy()]
    evaluations = 0
    for step_index in range(number_of_steps):
        time = start_time + step_index * step
        stage_times = time + tableau.nodes * step
        newton_factorisation = scipy.linalg.lu_factor(
            identity - step * numpy.kron(tableau.stage_matrix, jacobian(time, state)))
        stage_increments = numpy.outer(tableau.nodes, step * vector_field(time, state))
        evaluations += 1
        previous_correction_norm = numpy.inf
        for iteration in range(maximum_newton_iterations):
            stage_derivatives = vector_field(stage_times, (state + stage_increments).T).T
            evaluations += number_of_stages
            residual = step * tableau.stage_matrix @ stage_derivatives - stage_increments
            correction = scipy.linalg.lu_solve(newton_factorisation, residual.ravel()).reshape(number_of_stages, dimension)
            stage_increments += correction
            correction_norm = numpy.max(numpy.abs(correction))
            if correction_norm <= 4e-16 * (1 + numpy.max(numpy.abs(state))) or correction_norm >= previous_correction_norm:
                break
            previous_correction_norm = correction_norm
        stage_derivatives = vector_field(stage_times, (state + stage_increments).T).T
        evaluations += number_of_stages
        state = state + step * (tableau.weights @ stage_derivatives)
        if (step_index + 1) % record_stride == 0 or step_index == number_of_steps - 1:
            recorded_times.append(start_time + (step_index + 1) * step)
            recorded_states.append(state.copy())
    recorded_states = numpy.array(recorded_states)
    return OscillatorSolution(method=tableau.name, time=numpy.array(recorded_times), position=recorded_states[:, 0],
                              velocity=recorded_states[:, 1], accepted_steps=number_of_steps,
                              function_evaluations=evaluations, wall_time=wallclock.perf_counter() - started)


def solve_oscillator(parameters, initial_position, initial_velocity, time_span, method='taylor', output_times=None,
                     tolerance=1e-14, number_of_steps=None, number_of_stages=3, decimal_digits=None, series_order=None):
    start_time, end_time = time_span
    if method == 'taylor':
        solution = TaylorSeriesIntegrator(parameters, tolerance=tolerance, series_order=series_order,
                                          decimal_digits=decimal_digits).integrate(
            initial_position, initial_velocity, start_time, end_time, number_of_steps=number_of_steps)
        if output_times is not None:
            sampled_position, sampled_velocity = solution(output_times)
            solution.time, solution.position, solution.velocity = numpy.asarray(output_times), sampled_position, sampled_velocity
        return solution
    initial_state = numpy.array([initial_position, initial_velocity], dtype=float)
    if method == 'dormand-prince':
        return integrate_explicit_runge_kutta(parameters.vector_field, initial_state, start_time, end_time,
                                              dormand_prince_tableau, number_of_steps=number_of_steps,
                                              relative_tolerance=tolerance, absolute_tolerance=tolerance,
                                              output_times=output_times, record_every_step=output_times is None)
    if method == 'runge-kutta-4':
        return integrate_explicit_runge_kutta(parameters.vector_field, initial_state, start_time, end_time,
                                              classical_runge_kutta_tableau, number_of_steps=number_of_steps,
                                              output_times=output_times, record_every_step=output_times is None)
    if method == 'gauss-legendre':
        solution = integrate_gauss_legendre(parameters.vector_field, parameters.jacobian, initial_state, start_time,
                                            end_time, number_of_stages, number_of_steps)
        solution.method = 'Gauss-Legendre (own)'
        return solution
    if method.startswith('scipy-'):
        started = wallclock.perf_counter()
        scipy_result = scipy.integrate.solve_ivp(parameters.vector_field, (start_time, end_time), initial_state,
                                                 method=method.removeprefix('scipy-'), t_eval=output_times,
                                                 rtol=max(tolerance, 2.3e-14), atol=tolerance,
                                                 **({'jac': parameters.jacobian} if method in ('scipy-Radau', 'scipy-BDF', 'scipy-LSODA') else {}))
        return OscillatorSolution(method='SciPy ' + method.removeprefix('scipy-'), time=scipy_result.t,
                                  position=scipy_result.y[0], velocity=scipy_result.y[1],
                                  accepted_steps=scipy_result.t.size - 1 if output_times is None else 0,
                                  function_evaluations=scipy_result.nfev, wall_time=wallclock.perf_counter() - started)
    raise ValueError(f'unknown method {method}')


def locate_velocity_zeros(taylor_solution, newton_iterations=40):
    step_start_times, step_sizes, position_table, velocity_table = taylor_solution.series_segments
    derivative_table = velocity_table[:, 1:] * numpy.arange(1, velocity_table.shape[1])
    velocity_at_step_end = TaylorSeriesIntegrator.evaluate_series(velocity_table, step_sizes)
    crossing_steps = numpy.nonzero(velocity_table[:, 0] * velocity_at_step_end < 0)[0]
    resolution = mpmath.mpf(10) ** (4 - mpmath.mp.dps) if velocity_table.dtype == object else 4 * numpy.finfo(float).eps
    zero_times = []
    for step_index in crossing_steps:
        offset = step_sizes[step_index] * velocity_table[step_index, 0] / (velocity_table[step_index, 0] - velocity_at_step_end[step_index])
        for iteration in range(newton_iterations):
            correction = (TaylorSeriesIntegrator.evaluate_series(velocity_table[step_index], offset)
                          / TaylorSeriesIntegrator.evaluate_series(derivative_table[step_index], offset))
            offset = offset - correction
            if abs(correction) <= resolution * abs(step_sizes[step_index]):
                break
        zero_times.append(step_start_times[step_index] + offset)
    return numpy.array(zero_times, dtype=velocity_table.dtype)


def phase_space_derivatives(parameters, phase, components):
    position, velocity = components[0], components[1]
    time_per_phase = 1.0 / parameters.forcing_frequency
    derivatives = [velocity * time_per_phase,
                   (parameters.damping_coefficient * velocity + parameters.natural_frequency**2 * position
                    - parameters.cubic_stiffness * position**3
                    + parameters.forcing_amplitude * numpy.cos(phase)) * time_per_phase]
    local_stiffness = parameters.natural_frequency**2 - 3 * parameters.cubic_stiffness * position**2
    for tangent_index in range(2, len(components), 2):
        tangent_position, tangent_velocity = components[tangent_index], components[tangent_index + 1]
        derivatives.append(tangent_velocity * time_per_phase)
        derivatives.append((local_stiffness * tangent_position
                            + parameters.damping_coefficient * tangent_velocity) * time_per_phase)
    return derivatives


def advance_ensemble(parameters, phase, components, phase_step, number_of_steps):
    half_step, sixth_step = phase_step / 2, phase_step / 6
    for step_index in range(number_of_steps):
        first_slope = phase_space_derivatives(parameters, phase, components)
        second_slope = phase_space_derivatives(parameters, phase + half_step,
                                               [value + half_step * slope for value, slope in zip(components, first_slope)])
        third_slope = phase_space_derivatives(parameters, phase + half_step,
                                              [value + half_step * slope for value, slope in zip(components, second_slope)])
        fourth_slope = phase_space_derivatives(parameters, phase + phase_step,
                                               [value + phase_step * slope for value, slope in zip(components, third_slope)])
        components = [value + sixth_step * (first + 2 * second + 2 * third + fourth)
                      for value, first, second, third, fourth in
                      zip(components, first_slope, second_slope, third_slope, fourth_slope)]
        phase = phase + phase_step
    return phase, components


def lyapunov_spectrum(parameters, initial_positions, initial_velocities, transient_periods=100, measured_periods=400,
                      steps_per_period=128):
    initial_positions = numpy.asarray(initial_positions, dtype=float)
    initial_velocities = numpy.asarray(initial_velocities, dtype=float)
    phase_step = 2 * numpy.pi / steps_per_period
    phase, components = advance_ensemble(parameters, 0.0, [initial_positions, initial_velocities], phase_step,
                                         transient_periods * steps_per_period)
    phase = phase % (2 * numpy.pi)
    ones, zeros = numpy.ones_like(initial_positions), numpy.zeros_like(initial_positions)
    components = components[:2] + [ones, zeros, zeros, ones.copy()]
    accumulated_logarithms = numpy.zeros((2,) + initial_positions.shape)
    for period_index in range(measured_periods):
        phase, components = advance_ensemble(parameters, phase, components, phase_step, steps_per_period)
        phase = phase % (2 * numpy.pi)
        position, velocity, first_tangent_position, first_tangent_velocity, second_tangent_position, second_tangent_velocity = components
        first_norm = numpy.hypot(first_tangent_position, first_tangent_velocity)
        first_tangent_position, first_tangent_velocity = first_tangent_position / first_norm, first_tangent_velocity / first_norm
        projection = second_tangent_position * first_tangent_position + second_tangent_velocity * first_tangent_velocity
        second_tangent_position = second_tangent_position - projection * first_tangent_position
        second_tangent_velocity = second_tangent_velocity - projection * first_tangent_velocity
        second_norm = numpy.hypot(second_tangent_position, second_tangent_velocity)
        accumulated_logarithms[0] += numpy.log(first_norm)
        accumulated_logarithms[1] += numpy.log(second_norm)
        components = [position, velocity, first_tangent_position, first_tangent_velocity,
                      second_tangent_position / second_norm, second_tangent_velocity / second_norm]
    return accumulated_logarithms / (measured_periods * 2 * numpy.pi / parameters.forcing_frequency)


conservative_parameters = OscillatorParameters(damping_ratio=0.0, natural_frequency=1.0, cubic_stiffness=1.0)


def verify_accuracy_against_exact_solutions():
    orbit_amplitudes = {'intrawell dn orbit': '1.2', 'interwell cn orbit': '1.7'}

    for orbit_name, amplitude_text in orbit_amplitudes.items():
        with mpmath.workdps(64):
            high_precision_solution = solve_oscillator(conservative_parameters, amplitude_text, 0, (0, 20), method='taylor',
                                                       tolerance=mpmath.mpf(10)**-62, decimal_digits=64)
            exact_positions, exact_velocities, exact_period = exact_conservative_solution(
                conservative_parameters, mpmath.mpf(amplitude_text), [20], 64)
            high_precision_error = float(max(abs(high_precision_solution.position[-1] - exact_positions[0]),
                                             abs(high_precision_solution.velocity[-1] - exact_velocities[0])))
        record_verification(f'64-digit Taylor integration vs 64-digit Jacobi elliptic solution at t = 20 ({orbit_name})',
                            high_precision_error, 1e-55, high_precision_error < 1e-55)

        with mpmath.workdps(40):
            event_solution = solve_oscillator(conservative_parameters, amplitude_text, 0, (0, 30), method='taylor',
                                              tolerance=mpmath.mpf(10)**-38, decimal_digits=40)
            velocity_zero_times = locate_velocity_zeros(event_solution)
            measured_period = 2 * (velocity_zero_times[-1] - velocity_zero_times[0]) / (len(velocity_zero_times) - 1)
            exact_period = exact_conservative_solution(conservative_parameters, mpmath.mpf(amplitude_text), [0], 40)[2]
            period_error = float(abs(measured_period / exact_period - 1))
        record_verification(f'Period from Taylor dense-output event location vs 4K(m)/lambda or 2K(m)/lambda ({orbit_name}, 40 digits)',
                            period_error, 1e-33, period_error < 1e-33)

        float_amplitude = float(amplitude_text)
        double_precision_solution = solve_oscillator(conservative_parameters, float_amplitude, 0.0, (0.0, 30.0), method='taylor',
                                                     tolerance=1e-16)
        random_query_times = numpy.sort(numpy.random.default_rng(7).uniform(0, 30, 500))
        dense_positions, dense_velocities = double_precision_solution(random_query_times)
        exact_positions, exact_velocities, exact_period = exact_conservative_solution(conservative_parameters, float_amplitude, random_query_times)
        dense_output_error = float(max(numpy.max(numpy.abs(dense_positions - exact_positions)),
                                       numpy.max(numpy.abs(dense_velocities - exact_velocities))))
        record_verification(f'Taylor dense output at 500 off-grid times vs exact solution ({orbit_name})',
                            dense_output_error, 1e-13, dense_output_error < 1e-13)

        float_zero_times = locate_velocity_zeros(double_precision_solution)
        float_period = 2 * (float_zero_times[-1] - float_zero_times[0]) / (len(float_zero_times) - 1)
        float_period_error = abs(float_period / exact_period - 1)
        record_verification(f'Period from double-precision event location vs elliptic-integral period ({orbit_name})',
                            float_period_error, 1e-13, float_period_error < 1e-13)

        exact_final_position, exact_final_velocity, exact_period = exact_conservative_solution(
            conservative_parameters, float_amplitude, [30.0])
        for method_name, method_options, threshold in [
                ('taylor', {'tolerance': 1e-16}, 1e-13),
                ('dormand-prince', {'tolerance': 1e-12}, 1e-9),
                ('gauss-legendre', {'number_of_steps': 600, 'number_of_stages': 4}, 1e-11),
                ('runge-kutta-4', {'number_of_steps': 3000}, 1e-6),
                ('scipy-DOP853', {'tolerance': 1e-13}, 1e-10)]:
            candidate = solve_oscillator(conservative_parameters, float_amplitude, 0.0, (0.0, 30.0), method=method_name, **method_options)
            final_error = float(max(abs(candidate.position[-1] - exact_final_position[0]),
                                    abs(candidate.velocity[-1] - exact_final_velocity[0])))
            record_verification(f'{candidate.method} vs exact solution at t = 30 ({orbit_name})',
                                final_error, threshold, final_error < threshold)


def verify_convergence_orders():
    convergence_final_time, checkpoint_count = 10, 10
    convergence_studies = [
        ('Classical Runge-Kutta 4', 4, 'runge-kutta-4', {}, [240, 320, 480, 640, 960, 1280, 1920, 2560], None),
        ('Dormand-Prince 5 (fixed step)', 5, 'dormand-prince', {}, [120, 160, 240, 320, 480, 640, 960, 1280], None),
        ('Gauss-Legendre 1 stage (implicit midpoint)', 2, 'gauss-legendre', {'number_of_stages': 1},
         [240, 320, 480, 640, 960, 1280, 1920, 2560], None),
        ('Gauss-Legendre 2 stages', 4, 'gauss-legendre', {'number_of_stages': 2}, [60, 80, 120, 160, 240, 320, 480, 640], None),
        ('Gauss-Legendre 3 stages', 6, 'gauss-legendre', {'number_of_stages': 3}, [40, 60, 80, 120, 160, 240, 320], None),
        ('Gauss-Legendre 4 stages', 8, 'gauss-legendre', {'number_of_stages': 4}, [30, 40, 60, 80, 120, 160], None),
        ('Taylor series order 4', 4, 'taylor', {'series_order': 4}, [320, 480, 640, 960, 1280, 1920, 2560, 3840, 5120], None),
        ('Taylor series order 8 (60-digit arithmetic)', 8, 'taylor', {'series_order': 8}, [640, 960, 1280, 1920, 2560], 60),
        ('Taylor series order 12 (60-digit arithmetic)', 12, 'taylor', {'series_order': 12}, [320, 480, 640, 960, 1280], 60),
        ('Taylor series order 16 (60-digit arithmetic)', 16, 'taylor', {'series_order': 16}, [320, 480, 640, 960, 1280], 60),
    ]
    convergence_table = []
    for study_name, theoretical_order, method_name, method_options, step_counts, decimal_digits in convergence_studies:
        precision_context = mpmath.workdps(decimal_digits) if decimal_digits else contextlib.nullcontext()
        error_floor = 10.0 ** (8 - decimal_digits) if decimal_digits else 1e-12
        checkpoint_errors = []
        with precision_context:
            for step_count in step_counts:
                candidate = solve_oscillator(conservative_parameters, '1.7' if decimal_digits else 1.7, 0.0,
                                             (0, convergence_final_time), method=method_name, number_of_steps=step_count,
                                             decimal_digits=decimal_digits, **method_options)
                checkpoint_indices = numpy.arange(1, checkpoint_count + 1) * step_count // checkpoint_count
                checkpoint_times = candidate.time[checkpoint_indices]
                exact_positions, exact_velocities, exact_period = exact_conservative_solution(
                    conservative_parameters, '1.7' if decimal_digits else 1.7, checkpoint_times, decimal_digits)
                checkpoint_errors.append(float(max(max(abs(numerical - exact) for numerical, exact in
                                                       zip(candidate.position[checkpoint_indices], exact_positions)),
                                                   max(abs(numerical - exact) for numerical, exact in
                                                       zip(candidate.velocity[checkpoint_indices], exact_velocities)))))
        checkpoint_errors = numpy.array(checkpoint_errors)
        step_sizes = convergence_final_time / numpy.array(step_counts, dtype=float)
        usable = (checkpoint_errors > error_floor) & (checkpoint_errors < 1e-3)
        measured_order = numpy.polyfit(numpy.log(step_sizes[usable]), numpy.log(checkpoint_errors[usable]), 1)[0]
        convergence_table.append((study_name, theoretical_order, measured_order, int(usable.sum()),
                                  checkpoint_errors[usable].max(), checkpoint_errors[usable].min()))
        record_verification(f'Observed order of convergence: {study_name} (theory {theoretical_order})',
                            measured_order, f'{theoretical_order} +/- 0.25',
                            abs(measured_order - theoretical_order) < 0.25 and usable.sum() >= 4)

    print(f'{"integrator":<46} {"theory":>7} {"observed":>9} {"points":>7} {"error range fitted":>24}')
    for study_name, theoretical_order, measured_order, fitted_points, largest_error, smallest_error in convergence_table:
        print(f'{study_name:<46} {theoretical_order:>7d} {measured_order:>9.3f} {fitted_points:>7d} '
              f'{largest_error:>11.2e} .. {smallest_error:.2e}')


def verify_structure_preservation():
    long_time_amplitude, long_time_final_time, long_time_steps = 1.7, 2000.0, 20000
    initial_energy = conservative_parameters.energy(long_time_amplitude, 0.0)
    drift_ratios, maximum_energy_errors = {}, {}
    for method_name, method_options in [('gauss-legendre', {'number_of_stages': 2}), ('runge-kutta-4', {})]:
        long_run = solve_oscillator(conservative_parameters, long_time_amplitude, 0.0, (0.0, long_time_final_time),
                                    method=method_name, number_of_steps=long_time_steps, **method_options)
        energy_errors = numpy.abs(conservative_parameters.energy(long_run.position, long_run.velocity) - initial_energy)
        half_index = len(energy_errors) // 2
        drift_ratios[method_name] = energy_errors[half_index:].max() / energy_errors[:half_index].max()
        maximum_energy_errors[method_name] = energy_errors.max()
    record_verification('Symplectic Gauss-Legendre: energy error bounded over 315 periods (late/early maximum ratio)',
                        drift_ratios['gauss-legendre'], '< 1.25', drift_ratios['gauss-legendre'] < 1.25)
    record_verification('Classical Runge-Kutta 4 at the same step: energy error drifts secularly (late/early maximum ratio)',
                        drift_ratios['runge-kutta-4'], '> 1.6', drift_ratios['runge-kutta-4'] > 1.6)

    forward_run = solve_oscillator(conservative_parameters, 1.7, 0.0, (0.0, 40.0), method='taylor', tolerance=1e-16)
    backward_run = solve_oscillator(conservative_parameters, float(forward_run.position[-1]), float(forward_run.velocity[-1]),
                                    (40.0, 0.0), method='taylor', tolerance=1e-16)
    reversibility_error = float(max(abs(backward_run.position[-1] - 1.7), abs(backward_run.velocity[-1])))
    record_verification('Time reversibility: Taylor integration 0 -> 40 -> 0 returns to the initial state',
                        reversibility_error, 1e-12, reversibility_error < 1e-12)

    forced_parameters = OscillatorParameters(damping_ratio=-0.125, natural_frequency=1.0, cubic_stiffness=1.0,
                                             forcing_amplitude=0.3, forcing_frequency=1.0)

    def augmented_vector_field(time, state):
        return numpy.array([state[1], forced_parameters.acceleration(time, state[0], state[1]),
                            forced_parameters.input_power(time, state[0], state[1])])

    power_run = integrate_explicit_runge_kutta(augmented_vector_field, [0.1, 0.0, 0.0], 0.0, 300.0, dormand_prince_tableau,
                                               relative_tolerance=1e-12, absolute_tolerance=1e-12)
    energy_balance_residual = float(numpy.max(numpy.abs(
        forced_parameters.energy(power_run.position, power_run.velocity) - forced_parameters.energy(0.1, 0.0)
        - power_run.states[:, 2])))
    record_verification('Work-energy theorem H(t) - H(0) = integral of input power along a chaotic forced trajectory',
                        energy_balance_residual, 1e-8, energy_balance_residual < 1e-8)

    taylor_chaotic = solve_oscillator(forced_parameters, 0.1, 0.0, (0.0, 20.0), method='taylor', tolerance=1e-16)
    reference_chaotic = solve_oscillator(forced_parameters, 0.1, 0.0, (0.0, 20.0), method='scipy-DOP853', tolerance=1e-14)
    cross_check_error = float(max(abs(taylor_chaotic.position[-1] - reference_chaotic.position[-1]),
                                  abs(taylor_chaotic.velocity[-1] - reference_chaotic.velocity[-1])))
    record_verification('Own Taylor solver vs SciPy DOP853 on the forced chaotic regime at t = 20',
                        cross_check_error, 1e-9, cross_check_error < 1e-9)


def verify_chaos_diagnostics():
    chaotic_parameters = OscillatorParameters(damping_ratio=-0.125, natural_frequency=1.0, cubic_stiffness=1.0,
                                              forcing_amplitude=0.3, forcing_frequency=1.0)
    melnikov_threshold = float(melnikov_threshold_function(chaotic_parameters.damping_ratio, chaotic_parameters.natural_frequency,
                                                           chaotic_parameters.cubic_stiffness, chaotic_parameters.forcing_frequency))
    generator = numpy.random.default_rng(11)
    initial_positions, initial_velocities = generator.uniform(-1.5, 1.5, 16), generator.uniform(-1.0, 1.0, 16)
    expected_exponent_sum = 2 * chaotic_parameters.damping_ratio * chaotic_parameters.natural_frequency
    for forcing_amplitude, regime_name in [(0.3, 'above'), (0.5 * melnikov_threshold, 'below')]:
        regime_parameters = chaotic_parameters.replace(forcing_amplitude=forcing_amplitude)
        exponents = lyapunov_spectrum(regime_parameters, initial_positions, initial_velocities,
                                      transient_periods=100, measured_periods=400)
        largest_exponent, smallest_exponent = exponents[0].mean(), exponents[1].mean()
        liouville_error = float(numpy.max(numpy.abs(exponents.sum(axis=0) - expected_exponent_sum)))
        print(f'F = {forcing_amplitude:.4f} ({regime_name} Melnikov threshold F_c = {melnikov_threshold:.4f}): '
              f'lambda_1 = {largest_exponent:+.5f} +/- {exponents[0].std():.5f}, '
              f'lambda_2 = {smallest_exponent:+.5f}, Kaplan-Yorke dimension = '
              f'{2 + largest_exponent / abs(smallest_exponent) if largest_exponent > 0 else 1.0:.4f}')
        record_verification(f'Lyapunov spectrum sum equals Liouville divergence 2*eta*omega_n (F {regime_name} threshold)',
                            liouville_error, 1e-6, liouville_error < 1e-6)
        if regime_name == 'above':
            record_verification('Largest Lyapunov exponent positive above the Melnikov threshold (chaos)',
                                float(exponents[0].min()), '> 0.05', exponents[0].min() > 0.05)
        else:
            record_verification('Largest Lyapunov exponent negative below the Melnikov threshold (periodic attractor)',
                                float(exponents[0].max()), '< 0', exponents[0].max() < 0)


def load_notebook_namespace(notebook_path):
    notebook = json.loads(pathlib.Path(notebook_path).read_text())
    notebook_namespace = {}
    code_sources = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
    with contextlib.redirect_stdout(io.StringIO()):
        for code_source in code_sources:
            exec(compile(code_source, str(notebook_path), 'exec'), notebook_namespace)
    return notebook, code_sources, notebook_namespace


def verify_runge_kutta_notebook(notebook_path):
    notebook, code_sources, notebook_namespace = load_notebook_namespace(notebook_path)
    comment_count = sum(token.type == tokenize.COMMENT for code_source in code_sources
                        for token in tokenize.generate_tokens(io.StringIO(code_source).readline))
    non_code_cells = sum(cell['cell_type'] != 'code' for cell in notebook['cells'])
    record_verification('RK4 notebook: number of markdown cells plus comments', non_code_cells + comment_count, 0,
                        non_code_cells + comment_count == 0)
    solve_with_runge_kutta_4 = notebook_namespace['solve_with_runge_kutta_4']

    for orbit_name, amplitude in [('intrawell dn orbit', 1.2), ('interwell cn orbit', 1.7)]:
        notebook_times, notebook_positions, notebook_velocities = solve_with_runge_kutta_4(
            amplitude, 0.0, 0.0, 30.0, 0.001, 0.0, 1.0, 1.0)
        exact_positions, exact_velocities, exact_period = exact_conservative_solution(conservative_parameters, amplitude, notebook_times)
        notebook_error = float(max(numpy.max(numpy.abs(notebook_positions - exact_positions)),
                                   numpy.max(numpy.abs(notebook_velocities - exact_velocities))))
        record_verification(f'RK4 notebook vs Jacobi elliptic exact solution over 0 <= t <= 30, step 0.001 ({orbit_name})',
                            notebook_error, 1e-10, notebook_error < 1e-10)

    step_sizes = numpy.array([0.05, 0.025, 0.0125, 0.00625, 0.003125])
    checkpoint_times = numpy.arange(1.0, 11.0)
    exact_checkpoint_positions, exact_checkpoint_velocities, exact_period = exact_conservative_solution(
        conservative_parameters, 1.7, checkpoint_times)
    checkpoint_errors = []
    for step_size in step_sizes:
        notebook_times, notebook_positions, notebook_velocities = solve_with_runge_kutta_4(1.7, 0.0, 0.0, 10.0, step_size, 0.0, 1.0, 1.0)
        checkpoint_indices = numpy.rint(checkpoint_times / step_size).astype(int)
        checkpoint_errors.append(max(numpy.max(numpy.abs(notebook_positions[checkpoint_indices] - exact_checkpoint_positions)),
                                     numpy.max(numpy.abs(notebook_velocities[checkpoint_indices] - exact_checkpoint_velocities))))
    observed_order = numpy.polyfit(numpy.log(step_sizes), numpy.log(checkpoint_errors), 1)[0]
    record_verification('RK4 notebook: observed order of convergence against the exact solution', observed_order,
                        '4 +/- 0.15', abs(observed_order - 4) < 0.15)

    for damping_ratio_value in (-0.1, 0.05):
        notebook_times, notebook_positions, notebook_velocities = solve_with_runge_kutta_4(
            0.5, 0.3, 0.0, 30.0, 0.001, damping_ratio_value, 1.3, 0.8)
        reference = solve_oscillator(OscillatorParameters(damping_ratio=damping_ratio_value, natural_frequency=1.3,
                                                          cubic_stiffness=0.8), 0.5, 0.3, (0.0, 30.0), method='taylor',
                                     tolerance=1e-16, output_times=notebook_times[::100])
        reference_scale = 1 + max(numpy.max(numpy.abs(reference.position)), numpy.max(numpy.abs(reference.velocity)))
        damped_error = float(max(numpy.max(numpy.abs(notebook_positions[::100] - reference.position)),
                                 numpy.max(numpy.abs(notebook_velocities[::100] - reference.velocity))) / reference_scale)
        record_verification(f'RK4 notebook vs Taylor reference, eta = {damping_ratio_value:+.2f}, omega_n = 1.3, beta = 0.8 (relative)',
                            damped_error, 1e-9, damped_error < 1e-9)

    default_parameters = OscillatorParameters(damping_ratio=notebook_namespace['damping_ratio'],
                                              natural_frequency=notebook_namespace['natural_frequency'],
                                              cubic_stiffness=notebook_namespace['cubic_stiffness'])
    default_reference = solve_oscillator(default_parameters, notebook_namespace['initial_position'],
                                         notebook_namespace['initial_velocity'],
                                         (notebook_namespace['start_time'], notebook_namespace['end_time']),
                                         method='taylor', tolerance=1e-16, output_times=notebook_namespace['times'][::1000])
    default_scale = 1 + numpy.max(numpy.abs(default_reference.velocity))
    default_error = float(max(numpy.max(numpy.abs(notebook_namespace['positions'][::1000] - default_reference.position)),
                              numpy.max(numpy.abs(notebook_namespace['velocities'][::1000] - default_reference.velocity))) / default_scale)
    record_verification('RK4 notebook default run vs Taylor reference (relative to velocity scale)',
                        default_error, 1e-9, default_error < 1e-9)
    default_power = default_parameters.damping_coefficient * notebook_namespace['velocities']**2
    integrated_power = numpy.concatenate([[0.0], numpy.cumsum(
        (default_power[1:] + default_power[:-1]) / 2 * numpy.diff(notebook_namespace['times']))])
    power_balance_error = float(numpy.max(numpy.abs(notebook_namespace['energies'] - notebook_namespace['energies'][0]
                                                    - integrated_power)) / numpy.max(numpy.abs(notebook_namespace['energies'])))
    record_verification('RK4 notebook default run: energy change equals integrated 2*eta*omega_n*xdot^2 (trapezoid, relative)',
                        power_balance_error, 1e-6, power_balance_error < 1e-6)

    default_time_span = (notebook_namespace['start_time'], notebook_namespace['end_time'])
    print(f'Notebook default problem: eta = {default_parameters.damping_ratio}, omega_n = {default_parameters.natural_frequency}, '
          f'beta = {default_parameters.cubic_stiffness}, x(0) = {notebook_namespace["initial_position"]}, '
          f'xdot(0) = {notebook_namespace["initial_velocity"]}, t in {default_time_span}')
    print(f'{"method":<34} {"x(T)":>24} {"|x(T) - Taylor|":>16} {"steps":>7} {"f-evals":>8} {"seconds":>8}')
    taylor_final_position = None
    for method_name, method_options in [('taylor', {'tolerance': 1e-16}),
                                        ('dormand-prince', {'tolerance': 1e-12}),
                                        ('gauss-legendre', {'number_of_steps': 5000, 'number_of_stages': 4}),
                                        ('runge-kutta-4', {'number_of_steps': 50000}),
                                        ('scipy-DOP853', {'tolerance': 1e-13}),
                                        ('scipy-LSODA', {'tolerance': 1e-12})]:
        comparison = solve_oscillator(default_parameters, notebook_namespace['initial_position'],
                                      notebook_namespace['initial_velocity'], default_time_span,
                                      method=method_name, **method_options)
        final_position = float(comparison.position[-1])
        taylor_final_position = final_position if taylor_final_position is None else taylor_final_position
        print(f'{comparison.method:<34} {final_position:24.16e} {abs(final_position - taylor_final_position):16.3e} '
              f'{comparison.accepted_steps:7d} {comparison.function_evaluations:8d} {comparison.wall_time:8.3f}')
    notebook_final_position = float(notebook_namespace['positions'][-1])
    print(f'{"RK4 notebook (step 0.001)":<34} {notebook_final_position:24.16e} '
          f'{abs(notebook_final_position - taylor_final_position):16.3e}')


def print_verification_report():
    description_width = max(len(description) for description, measured, threshold, passed in verification_records)
    print()
    print(f'{"":6}{"verification":<{description_width}}  {"measured":>14}  {"threshold":>14}')
    for description, measured, threshold, passed in verification_records:
        measured_text = f'{measured:.3e}' if isinstance(measured, float) else str(measured)
        threshold_text = f'{threshold:.0e}' if isinstance(threshold, float) else str(threshold)
        print(f'{"PASS" if passed else "FAIL":<6}{description:<{description_width}}  {measured_text:>14}  {threshold_text:>14}')
    passed_count = sum(passed for description, measured, threshold, passed in verification_records)
    print(f'\n{passed_count} of {len(verification_records)} verifications passed')
    return passed_count == len(verification_records)


def main():
    notebook_path = (pathlib.Path(sys.argv[1]) if len(sys.argv) > 1
                     else pathlib.Path(__file__).resolve().with_name('duffing_rk4_solver.ipynb'))
    verification_stages = [
        ('symbolic derivations', verify_symbolic_derivations),
        ('model consistency', verify_model_consistency),
        ('accuracy against exact solutions', verify_accuracy_against_exact_solutions),
        ('convergence orders', verify_convergence_orders),
        ('structure preservation', verify_structure_preservation),
        ('chaos diagnostics', verify_chaos_diagnostics),
        ('RK4 notebook', lambda: verify_runge_kutta_notebook(notebook_path)),
    ]
    for stage_name, stage_function in verification_stages:
        stage_started = wallclock.perf_counter()
        stage_function()
        print(f'[{stage_name}] finished in {wallclock.perf_counter() - stage_started:.1f} s')
    return 0 if print_verification_report() else 1


if __name__ == '__main__':
    sys.exit(main())
