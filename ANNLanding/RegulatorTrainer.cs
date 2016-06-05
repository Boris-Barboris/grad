using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNLanding
{

    class NeuralControlSystem : IControlSystem
    {
        TansigAnn regulator;
        RegulatorTrainer rt;

        public NeuralControlSystem(TansigAnn regulator, RegulatorTrainer rt)
        {
            this.regulator = regulator.Copy();
            this.rt = rt;
        }

        Matrix norm_state = new Matrix(7, 1);

        public Matrix ApplyControl(Matrix pure_state, double time)
        {
            Matrix output = new Matrix(3, 1);
            rt.getRegulInput(pure_state, norm_state);
            Matrix signal = regulator.eval(norm_state);
            output[2, 0] = Rescale.norm(signal[0, 0], -TansigAnn.tanh_factor, TansigAnn.tanh_factor);
            return output;
        }
    }

    class RegulatorTrainer
    {
        public RegulatorTrainer()
        {
            init_weight_targets();
        }

        public void init_weight_targets()
        {
            desired_end_state[0, 0] = Rescale.norm(0.0, Xmin, Xmax);
            desired_end_state[1, 0] = Rescale.norm(0.0, Ymin, Ymax);
            desired_end_state[3, 0] = Rescale.norm_input(-0.5, 1);
            desired_end_state[4, 0] = Rescale.norm_input(0.0, 2);

            // координаты
            end_state_weights[0, 0] = 10.0;
            end_state_weights[1, 0] = 1.0;
            end_state_weights[3, 0] = 2.0;
            end_state_weights[4, 0] = 1.0;

            transit_state_weights[3, 0] = 0.005;
            transit_state_weights[4, 0] = 0.005;
        }

        public static double Xmin = -200.0;
        public static double Xmax = 0.0;
        public static double Ymin = 0.0;
        public static double Ymax = 50.0;

        List<TansigAnn.TrainingProcess> training_set = new List<TansigAnn.TrainingProcess>();
        List<TansigAnn.TrainingProcess> generalization_set;
        List<TansigAnn.TrainingProcess> validation_set;

        public void generate_training_processes(int set_size, double x, double y, double dy, double alpha0, double cz,
            double vx, double vy, double gen_percent = 20.0, double valid_percent = 10.0, bool valid = true, double dt = 0.05)
        {
            training_set = new List<TansigAnn.TrainingProcess>(set_size);
            this.dt = dt;

            Random rnd = new Random();
            for (int i = 0; i < set_size; i++)
            {
                Matrix init_state = new Matrix(7, 1);
                init_state[0, 0] = Rescale.norm(x, Xmin, Xmax);
                if (set_size > 1)
                    init_state[1, 0] = Rescale.norm(y + (-0.5 + i * (1.0 / (set_size - 1))) * dy, Ymin, Ymax);
                else
                    init_state[1, 0] = Rescale.norm(y, Ymin, Ymax);
                init_state[2, 0] = Rescale.norm_input(vx, 0);
                init_state[3, 0] = Rescale.norm_input(vy, 1);
                init_state[4, 0] = Rescale.norm_input(0.0, 2);
                init_state[5, 0] = Rescale.norm_input(alpha0, 3);
                init_state[6, 0] = Rescale.norm_input(cz, 4);
                // TODO
                TansigAnn.TrainingProcess tr_pair = new TansigAnn.TrainingProcess(init_state, training_step_dlg, msqrerr);
                training_set.Add(tr_pair);
            }

            // Теперь разбить пары на три набора: тренировочный, генерализации и валидации
            int gen_count = 0;// (int)(set_size / 100.0 * gen_percent);
            int valid_count = 0;// (int)(set_size / 100.0 * valid_percent);

            generalization_set = new List<TansigAnn.TrainingProcess>();

            while (gen_count > 0)
            {
                int index = rnd.Next() % training_set.Count;
                generalization_set.Add(training_set[index]);
                training_set.RemoveAt(index);
                gen_count--;
            }

            if (valid)
            {
                validation_set = new List<TansigAnn.TrainingProcess>();
                while (valid_count > 0)
                {
                    int index = rnd.Next() % training_set.Count;
                    validation_set.Add(training_set[index]);
                    training_set.RemoveAt(index);
                    valid_count--;
                }
            }
        }

        public double dt = 0.05;
        public Matrix desired_end_state = new Matrix(7, 1);
        public Matrix desired_transit_state = new Matrix(7, 1);
        public Matrix end_state_weights = new Matrix(7, 1);
        public Matrix transit_state_weights = new Matrix(7, 1);

        double training_step_dlg(TansigAnn emulator, TansigAnn regulator,
                Matrix currentState, Matrix next_state, Matrix[] emulator_jacobians, Matrix[] control_jacobians, Matrix global_jacobian,
                Matrix err_gradient, out bool finished, int length)
        {
            // first let's calculate output of regulator
            Matrix signal = new Matrix(1, 1);
            int param_count = emulator_jacobians[0].cols;
            // control_jacobians first is pretty much copy of global jacobian
            global_jacobian.CopyTo(control_jacobians[0]);
            regulator.fwdpropagation_process(currentState, signal, control_jacobians);
            // clamp output
            signal[0, 0] = Rescale.norm(signal[0, 0], -TansigAnn.tanh_factor, TansigAnn.tanh_factor);

            // copy output layer control jacobian to last row of emulator jacobian
            Matrix lastjacob = control_jacobians[control_jacobians.Length - 1];
            for (int i = 0; i < lastjacob.cols; i++)
                emulator_jacobians[0][5, i] = Rescale.normDelta(lastjacob[0, i], -TansigAnn.tanh_factor, TansigAnn.tanh_factor);
            // emulator jacobian is partially global jacobian
            for (int i = 0; i < 5; i++)
                for (int p = 0; p < param_count; p++)
                    emulator_jacobians[0][i, p] = global_jacobian[i + 2, p];

            // now we can get emulator to work
            Matrix emul_input = getEmulInput(currentState, signal);
            Matrix emul_output = new Matrix(6, 1);
            emulator.fwdpropagation_derivs(emul_input, emul_output, emulator_jacobians);

            // calculate next state
            next_state[0, 0] = currentState[0, 0] + Rescale.normDelta(Rescale.pure_output(emul_output[0, 0], 0), Xmin, Xmax);
            next_state[1, 0] = currentState[1, 0] + Rescale.normDelta(Rescale.pure_output(emul_output[1, 0], 1), Ymin, Ymax);
            for (int i = 2; i < 6; i++)
                next_state[i, 0] = currentState[i, 0] + Rescale.norm_input_delta(Rescale.pure_output(emul_output[i, 0], i), i - 2);
            // actuator
            double raw_act1 = PlaneModel.csurf_actuator(currentState[6, 0], signal[0, 0], dt);
            next_state[6, 0] = raw_act1;

            // numerical partial derivatives of actuator position
            double raw_act2_state = PlaneModel.csurf_actuator(currentState[6, 0] + 1e-5, signal[0, 0], dt);
            double act_deriv_state = (raw_act2_state - raw_act1) / 1e-5;
            double raw_act2_input = PlaneModel.csurf_actuator(currentState[6, 0], signal[0, 0] + 1e-5, dt);
            double act_deriv_input = (raw_act2_input - raw_act1) / 1e-5;

            // last row of jacobians is only a part of jacobian
            Matrix emul_oj = emulator_jacobians[emulator_jacobians.Length - 1];
            Matrix new_jacob = new Matrix(7, param_count);
            
            // account for diffirence method
            for (int p = 0; p < param_count; p++)
            {
                new_jacob[0, p] = global_jacobian[0, p] + Rescale.normDelta(Rescale.pure_output_delta(emul_oj[0, p], 0), Xmin, Xmax);
                new_jacob[1, p] = global_jacobian[1, p] + Rescale.normDelta(Rescale.pure_output_delta(emul_oj[1, p], 1), Ymin, Ymax);
                for (int i = 2; i < 6; i++)
                    new_jacob[i, p] = global_jacobian[i, p] + Rescale.norm_input_delta(Rescale.pure_output_delta(emul_oj[i, p], i), i - 2);
                // also, numerical derivative for actuator
                new_jacob[6, p] = act_deriv_state * global_jacobian[6, p] + act_deriv_input *
                    Rescale.normDelta(lastjacob[0, p], -TansigAnn.tanh_factor, TansigAnn.tanh_factor);
            }

            // update first global jacobian with this
            new_jacob.CopyTo(global_jacobian);

            // let's see, if we already finished
            finished = check_finished(next_state);

            if (finished)
            {
                // calculate errors for finish state
                double error = 0.0;
                Matrix errors = state_errors(next_state, desired_end_state, end_state_weights);
                // calculate total error
                for (int i = 0; i < 7; i++)
                    error += errors[i, 0] * errors[i, 0];
                // divide gradient by length
                if (length > 0)
                    for (int p = 0; p < param_count; p++)
                        err_gradient[p, 0] /= length;
                // calculate gradient
                for (int p = 0; p < param_count; p++)
                {
                    for (int i = 0; i < 7; i++)
                    {
                        err_gradient[p, 0] -= 2.0 * end_state_weights[i, 0] * errors[i, 0] * new_jacob[i, p];
                    }
                }
                return error;
            }
            else
            {
                // calculate gradient for transient state
                double error = 0.0;
                Matrix errors = state_errors(next_state, desired_transit_state, transit_state_weights);
                // calculate transient error
                for (int i = 0; i < 7; i++)
                    error += errors[i, 0] * errors[i, 0] * dt;
                // calculate gradient
                for (int p = 0; p < param_count; p++)
                {
                    for (int i = 0; i < 7; i++)
                    {
                        err_gradient[p, 0] -= 2.0 * end_state_weights[i, 0] * errors[i, 0] * new_jacob[i, p] * dt;
                    }
                }
                return error;
            }
        }

        public Matrix getEmulInput(Matrix full_norm_state, Matrix signal)
        {
            Matrix emul_input = new Matrix(6, 1);
            for (int i = 0; i < 5; i++)
                emul_input[i, 0] = full_norm_state[i + 2, 0];
            emul_input[5, 0] = signal[0, 0];
            return emul_input;
        }

        public void getRegulInput(Matrix pure_state, Matrix norm_state)
        {
            norm_state[0, 0] = Rescale.norm(pure_state[0, 0], Xmin, Xmax);
            norm_state[1, 0] = Rescale.norm(pure_state[1, 0], Ymin, Ymax);
            for (int i = 2; i < 7; i++)
                norm_state[i, 0] = Rescale.norm_input(pure_state[i, 0], i - 2);
        }

        bool check_finished(Matrix full_state)
        {
            double x = full_state[0, 0];
            double y = Rescale.pure(full_state[1, 0], Ymin, Ymax);
            if (x >= desired_end_state[0, 0])
                return true;
            //if (y < 0.0)
            //    return true;
            //double vx = Rescale.pure_input(full_state[2, 0], 0);
            //double vy = Rescale.pure_input(full_state[3, 0], 1);
            //if (vx < Rescale.input_lower_bounds[0] || vx > Rescale.input_upper_bounds[0])
            //    return true;
            //if (vy < Rescale.input_lower_bounds[1] || vy > Rescale.input_upper_bounds[1])
            //    return true;
            return false;
        }

        Matrix state_errors(Matrix state, Matrix desired_state, Matrix error_weights)
        {
            Matrix errors = new Matrix(state.rows, 1);
            for (int i = 0; i < state.rows; i++)
            {
                errors[i, 0] = (desired_state[i, 0] - state[i, 0]) * error_weights[i, 0];
            }
            return errors;
        }

        double msqrerr(TansigAnn emulator, TansigAnn regulator, Matrix start_state, int length)
        {
            bool end = false;
            double err = 0.0;
            double err_finish = 0.0;
            int iter = 0;
            Matrix cur_state = start_state.Copy();
            Matrix next_state = new Matrix(7, 1);
            while (!end && iter < 1000)
            {
                // first let's calculate output of regulator
                Matrix signal = regulator.eval(cur_state);
                // clamp output
                signal[0, 0] = Rescale.norm(signal[0, 0], -TansigAnn.tanh_factor, TansigAnn.tanh_factor);
                
                // proceed with emulator
                Matrix emul_input = getEmulInput(cur_state, signal);
                Matrix delta = emulator.eval(emul_input);

                // let's apply delta
                next_state[0, 0] = cur_state[0, 0] + Rescale.normDelta(Rescale.pure_output(delta[0, 0], 0), Xmin, Xmax);
                next_state[1, 0] = cur_state[1, 0] + Rescale.normDelta(Rescale.pure_output(delta[1, 0], 1), Ymin, Ymax);
                for (int i = 2; i < 6; i++)
                    next_state[i, 0] = cur_state[i, 0] + Rescale.norm_input_delta(Rescale.pure_output(delta[i, 0], i), i - 2);
                // actuator
                double raw_act1 = PlaneModel.csurf_actuator(cur_state[6, 0], signal[0, 0], dt);
                next_state[6, 0] = PlaneModel.Clamp(raw_act1, -1.0, 1.0);

                // let's see, if we already finished
                if (length == 0)
                    end = check_finished(next_state);
                else
                    end = length == iter + 1;
                //end = check_finished(next_state);

                if (end)
                {
                    //calculate errors for finish state
                    Matrix errors = state_errors(next_state, desired_end_state, end_state_weights);
                    // calculate total error
                    for (int i = 0; i < 7; i++)
                        err_finish += errors[i, 0] * errors[i, 0];
                }
                else
                {
                    // calculate gradient for transient state
                    Matrix errors = state_errors(next_state, desired_transit_state, transit_state_weights);
                    // calculate transient error
                    for (int i = 0; i < 7; i++)
                        err += errors[i, 0] * errors[i, 0] * dt;
                }

                next_state.CopyTo(cur_state);
                iter++;
            }
            if (iter > 1)
                err /= iter - 1.0;
            return err + err_finish;
        }

        public void train_regulator_processes(TansigAnn regulator, TansigAnn emulator, Action<int, double, double, double> report_dlg, ref bool stop_flag,
            double speed)
        {
            // обнулим длины процессов
            foreach (var p in training_set)
                p.length = 0;
            foreach (var p in generalization_set)
                p.length = 0;
            foreach (var p in validation_set)
                p.length = 0;
            //regulator.train_CGBP_process(training_set, generalization_set, validation_set, emulator, 1e-8, 100000, report_dlg, ref stop_flag, 100, 30);
            regulator.train_BP_stochastic_process(training_set, generalization_set, validation_set, emulator, 1e-8, 100000, report_dlg, ref stop_flag, 100, 30, speed, 1.0, 1.0);
        }

        void rng_norm_state(Random rng, Matrix norm_state)
        {
            for (int i = 0; i < 7; i++)
            {
                norm_state[i, 0] = 2.0 * (rng.NextDouble() - 0.5);
            }
        }

        // научим регулятор всегда выдавать один выход на всём фазовом пространстве
        public void pretrain_regulator_dumb(TansigAnn regulator, Action<int, double, double, double> report_dlg, int set_size, double desired_output,
            int iter_limit, ref bool stop_flag)
        {
            List<TansigAnn.TrainingPair> training_set = new List<TansigAnn.TrainingPair>(set_size);
            Random rng = new Random();
            for (int i = 0; i < set_size; i++)
            {
                Matrix input = new Matrix(7, 1);
                rng_norm_state(rng, input);
                Matrix output = new Matrix(1, 1);
                output[0, 0] = Rescale.pure(desired_output, -TansigAnn.tanh_factor, TansigAnn.tanh_factor);
                training_set.Add(new TansigAnn.TrainingPair(input, output));
            }

            // пустышки
            Matrix err_weight = Matrix.IdentityMatrix(1, 1);

            // тренировка
            regulator.train_CGBP(training_set, new List<TansigAnn.TrainingPair>(), new List<TansigAnn.TrainingPair>(),
                err_weight, 7e-2, iter_limit, report_dlg, ref stop_flag, 100, 100);
        }

    }
}
