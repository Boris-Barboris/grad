using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.IO;

namespace ANNLanding
{
    public class TansigAnn
    {
        public TansigAnn(int inputs_count, int layer_count, int[] layer_sizes, bool purelin_output, 
            double weight_rng_span, double bias_rng_span, bool pos_weights = false)
        {
            this.inputs_count = inputs_count;
            this.layer_count = layer_count;
            weights = new Matrix[layer_count];
            biases = new Matrix[layer_count];
            outputs = new Matrix[layer_count];
            net_inputs = new Matrix[layer_count];
            this.purelin_output = purelin_output;
            // input layer
            weights[0] = new Matrix(layer_sizes[0], inputs_count);
            biases[0] = new Matrix(layer_sizes[0], 1);
            outputs[0] = new Matrix(layer_sizes[0], 1);
            net_inputs[0] = new Matrix(layer_sizes[0], 1);
            // other layers
            for (int i = 1; i < layer_count; i++)
            {
                weights[i] = new Matrix(layer_sizes[i], layer_sizes[i-1]);
                biases[i] = new Matrix(layer_sizes[i], 1);
                outputs[i] = new Matrix(layer_sizes[i], 1);
                net_inputs[i] = new Matrix(layer_sizes[i], 1);
            }
            // rng
            Random rnd = new Random();
            for (int i = 0; i < layer_count; i++)
            {
                double std_dev = 1.0 / Math.Sqrt(i == 0 ? inputs_count : layer_sizes[i - 1]);
                double span = 1.5 * std_dev;
                for (int r = 0; r < layer_sizes[i]; r++)
                {
                    for (int c = 0; c < weights[i].cols; c++)
                        weights[i][r, c] = (pos_weights ? rnd.NextDouble() : rnd.NextDouble() - 0.5) * 2.0 * weight_rng_span * span;
                    biases[i][r, 0] = (rnd.NextDouble() - 0.5) * 2.0 * bias_rng_span * span;
                }
            }
        }

        public TansigAnn Copy()
        {
            Matrix[] copy_weights = new Matrix[weights.Length];
            Matrix[] copy_biases = new Matrix[biases.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                copy_weights[i] = weights[i].Copy();
                copy_biases[i] = biases[i].Copy();
            }
            TansigAnn copy = new TansigAnn(copy_weights, copy_biases, inputs_count, layer_count, purelin_output);
            return copy;
        }

        public readonly int inputs_count;
        public readonly int layer_count;

        public readonly bool purelin_output;

        public Matrix[] weights;
        public Matrix[] biases;
        public Matrix[] outputs;
        public Matrix[] net_inputs;

        /// <summary>
        /// Save ANN to text file.
        /// </summary>
        /// <returns>true if successfull</returns>
        public void Serialize(string filename)
        {
            using (StreamWriter writer = new StreamWriter(filename))
            {
                writer.WriteLine("inputs_count= " + inputs_count.ToString());
                writer.WriteLine("layer_count= " + layer_count.ToString());
                writer.WriteLine("purelin_output= " + purelin_output.ToString());
                for (int i = 0; i < weights.Length; i++)
                {
                    writer.WriteLine("weights" + i.ToString());
                    writer.WriteLine(weights[i].ToString());
                }
                for (int i = 0; i < biases.Length; i++)
                {
                    writer.WriteLine("biases" + i.ToString());
                    writer.WriteLine(biases[i].ToString());
                }
            }
        }

        /// <summary>
        /// Load ANN from text file.
        /// </summary>
        /// <param name="filename"></param>
        /// <returns></returns>
        public static TansigAnn Deserialize(string filename)
        {
            using (StreamReader reader = new StreamReader(filename))
            {
                int inputs_count = int.Parse(reader.ReadLine().Split(' ')[1]);
                int layer_count = int.Parse(reader.ReadLine().Split(' ')[1]);
                bool purelin_output = bool.Parse(reader.ReadLine().Split(' ')[1]);
                Matrix[] weights = new Matrix[layer_count];
                Matrix[] biases = new Matrix[layer_count];
                StringBuilder builder = new StringBuilder(1000);
                // weights
                for (int i = 0; i < layer_count; i++)
                {
                    builder.Clear();
                    reader.ReadLine();
                    string str = reader.ReadLine();
                    while (str != "\r\n" && str.Length > 0)
                    {
                        builder.Append(str + "\r\n");
                        str = reader.ReadLine();
                    }
                    weights[i] = Matrix.Parse(builder.ToString());
                }
                // biases
                for (int i = 0; i < layer_count; i++)
                {
                    builder.Clear();
                    reader.ReadLine();
                    string str = reader.ReadLine();
                    while (str != "\r\n" && str.Length > 0)
                    {
                        builder.Append(str + "\r\n");
                        str = reader.ReadLine();
                    }
                    biases[i] = Matrix.Parse(builder.ToString());
                }

                return new TansigAnn(weights, biases, inputs_count, layer_count, purelin_output);
            }
        }

        private TansigAnn(Matrix[] weights, Matrix[] biases, int inputs_count, int layer_count, bool purelin_output)
        {
            this.weights = weights;
            this.biases = biases;
            this.inputs_count = inputs_count;
            this.layer_count = layer_count;
            this.purelin_output = purelin_output;
            outputs = new Matrix[layer_count];
            net_inputs = new Matrix[layer_count];
            for (int i = 0; i < layer_count; i++)
            {
                outputs[i] = new Matrix(weights[i].rows, 1);
                net_inputs[i] = new Matrix(weights[i].rows, 1);
            }
        }

        /// <summary>
        /// First derivative of hyperbolic tangent
        /// </summary>
        public static double tanh_deriv(double x)
        {
            double tanh = Math.Tanh(x);
            return 1.0 - tanh * tanh;
        }

        public const double tanh_factor = 1.7159;
        const double net_factor = 2.0 / 3.0;

        public Matrix eval(Matrix input)
        {
            for (int i = 0; i < layer_count; i++)
            {
                Matrix.Multiply(weights[i], i == 0 ? input : outputs[i-1], ref net_inputs[i]);  // weights
                Matrix.Add(net_inputs[i], biases[i], ref net_inputs[i]);                        // biases
                if (i < layer_count - 1 || !purelin_output)
                {
                    for (int j = 0; j < outputs[i].rows; j++)
                        outputs[i][j, 0] = tanh_factor * Math.Tanh(net_factor * net_inputs[i][j, 0]);                      // tansig
                }
                else
                {
                    for (int j = 0; j < outputs[i].rows; j++)
                        outputs[i][j, 0] = net_inputs[i][j, 0];                                 // purelin
                }
            }
            return outputs[layer_count - 1];
        }

        public class TrainingPair
        {
            public TrainingPair(Matrix input, Matrix output)
            {
                this.input = input;
                this.correct_output = output;
            }
            public Matrix input;
            public Matrix correct_output;
        }

        public class TrainingProcess
        {
            public TrainingProcess(Matrix init_state, TimeStepDlg stepFunc, MeasureErrorDlg sqrerrFunc)
            {
                this.init_state = init_state;
                step = stepFunc;
                sqrerr = sqrerrFunc;
            }
            public Matrix init_state;
            public int length = 0;
            public List<Matrix> process = null;
            public delegate double TimeStepDlg(TansigAnn emulator, TansigAnn regulator,
                Matrix currentState, Matrix next_state, Matrix[] emul_jacobians, Matrix[] control_jacobians, 
                Matrix global_jacobian, Matrix err_gradient, out bool finished, int length);
            public delegate double MeasureErrorDlg(TansigAnn emulator, TansigAnn regulator, Matrix start_state, int length);
            public TimeStepDlg step;
            public MeasureErrorDlg sqrerr;
        }

        int param_count = 0;

        struct Particle
        {
            public TansigAnn local_ann;
            public double best_error;
            public double current_error;

            public Matrix best_position;
            public Matrix cur_position;
            public Matrix velocity;

            public Particle(double weight_span, double bias_span, TansigAnn original_ann, int param_count, 
                double max_v, Random rng, bool keep_training)
            {
                int[] layer_sizes = new int[original_ann.layer_count];
                for (int i = 0; i < original_ann.layer_count; i++)
                    layer_sizes[i] = original_ann.biases[i].rows;
                local_ann = new TansigAnn(original_ann.inputs_count, original_ann.layer_count, layer_sizes,
                    original_ann.purelin_output, weight_span, bias_span);
                if (keep_training)
                {
                    for (int l = 0; l < local_ann.layer_count; l++)
                    {
                        Matrix.Add(original_ann.weights[l], local_ann.weights[l], ref local_ann.weights[l]);
                        Matrix.Add(original_ann.biases[l], local_ann.biases[l], ref local_ann.biases[l]);
                    }
                }
                best_position = new Matrix(param_count, 1);
                cur_position = new Matrix(param_count, 1);
                velocity = new Matrix(param_count, 1);
                best_error = current_error = double.MaxValue;
                for (int i = 0; i < param_count; i++)
                    velocity[i, 0] = rng.NextDouble() * max_v;
            }

            public void eval_cur_error(IList<TrainingPair> training_set, Matrix error_weights)
            {
                current_error = local_ann.meansqr_err(training_set, error_weights);
                if (current_error < best_error)
                {
                    best_error = current_error;
                    local_ann.params_to_matrix(best_position);
                }
            }

            //public void eval_cur_error(IList<TrainingProcess> training_set, Matrix error_weights)
            //{
            //    current_error = local_ann.meansqr_err(training_set, error_weights);
            //    if (current_error < best_error)
            //    {
            //        best_error = current_error;
            //        local_ann.params_to_matrix(best_position);
            //    }
            //}

            public void move(Matrix global_best, Matrix beta1, Matrix beta2, double inertia, double c1, double c2, double max_v)
            {
                local_ann.params_to_matrix(cur_position);
                // alter velocity
                double max_v_abs = 0.0;
                for (int r = 0; r < velocity.rows; r++)
                {
                    velocity[r, 0] = inertia * velocity[r, 0] + c1 * beta1[r, 0] * (best_position[r, 0] - cur_position[r, 0]) +
                        c2 * beta2[r, 0] * (global_best[r, 0] - cur_position[r, 0]);
                    double abs_v_r = Math.Abs(velocity[r, 0]);
                    if (abs_v_r > max_v_abs)
                        max_v_abs = abs_v_r;
                }
                // clamp velocity
                if (max_v_abs > max_v)
                {
                    double factor = max_v / max_v_abs;
                    for (int r = 0; r < velocity.rows; r++)
                        velocity[r, 0] *= factor;
                }
                // move
                local_ann.add_to_params(velocity);
            }
        }

        public void tain_PSO(IList<TrainingPair> training_set, IList<TrainingPair> generalization_set,
            IList<TrainingPair> validation_set, Matrix error_weights, int particle_count, int iteration_count,
            Action<int, double, double, double> report_delegate, double inertia, double c1, double c2, ref bool stop_flag,
            double weight_span, double bias_span, double max_v, bool keep_training)
        {
            int iteration = 0;
            param_count = 0;
            for (int i = 0; i < layer_count; i++)
            {
                param_count += weights[i].rows * weights[i].cols;
                param_count += biases[i].rows;
            }
            // random speed vectors
            Matrix beta1 = new Matrix(param_count, 1);
            Matrix beta2 = new Matrix(param_count, 1);
            Random rng = new Random();
            // particles
            List<Particle> particles = new List<Particle>(particle_count);
            for (int i = 0; i < particle_count; i++)
            {
                if (keep_training && i == 0)
                {
                    // at least one particle at current position
                    particles.Add(new Particle(0.0, 0.0, this, param_count, max_v, rng, keep_training));
                }
                else
                    particles.Add(new Particle(weight_span, bias_span, this, param_count, max_v, rng, keep_training));
            }
            // best fit
            int best_particle = -1;
            double best_error = double.MaxValue;
            Matrix best_params = new Matrix(param_count, 1);

            // main algorithm
            do
            {
                if (iteration == 0)
                {
                    // initial report and initialization
                    for (int i = 0; i < particle_count; i++)
                    {
                        Particle p = particles[i];
                        p.eval_cur_error(training_set, error_weights);
                        if (p.best_error < best_error)
                        {
                            best_particle = i;
                            best_error = p.best_error;
                            best_params = p.best_position;
                        }
                        if (stop_flag)
                            break;
                    }
                    matrix_to_params(best_params);
                    report_delegate(iteration, best_error, meansqr_err(generalization_set, error_weights), 
                        meansqr_err(validation_set, error_weights));
                }

                // generate betas
                for (int i = 0; i < param_count; i++)
                {
                    beta1[i, 0] = rng.NextDouble();
                    beta2[i, 0] = rng.NextDouble();
                }
                // move
                for (int i = 0; i < particle_count; i++)
                    particles[i].move(best_params, beta1, beta2, inertia, c1, c2, max_v);
                // update everything
                bool found_better = false;
                for (int i = 0; i < particle_count; i++)
                {
                    Particle p = particles[i];
                    p.eval_cur_error(training_set, error_weights);
                    if (p.best_error < best_error)
                    {
                        best_particle = i;
                        best_error = p.best_error;
                        best_params = p.best_position;
                        found_better = true;
                    }
                    if (stop_flag)
                        break;
                }
                if (found_better)
                    matrix_to_params(best_params);
                // report
                report_delegate(iteration, best_error, meansqr_err(generalization_set, error_weights),
                    meansqr_err(validation_set, error_weights));
            } while (!stop_flag && iteration < iteration_count);
        }

        public void train_CGBP(IList<TrainingPair> training_set, IList<TrainingPair> generalization_set, 
            IList<TrainingPair> validation_set, Matrix error_weights,
			double convergence_epsilon, int iteration_limit, Action<int, double, double, double> report_delegate, ref bool stop_flag,
            int convergence_count_limit, int overlearning_count_limit)
        {
            // let's create required structures
            Matrix[] reserve_weights = new Matrix[layer_count];
            Matrix[] reserve_biases = new Matrix[layer_count];
            param_count = 0;
            for (int i = 0; i < layer_count; i++)
            {
                param_count += weights[i].rows * weights[i].cols;
                param_count += biases[i].rows;
                reserve_weights[i] = weights[i].Copy();
                reserve_biases[i] = biases[i].Copy();
            }
            Matrix avg_gradient_mat = new Matrix(param_count, 1);
            Matrix prev_gradient_mat = new Matrix(param_count, 1);
            Matrix prev_cunjugate_gradient = new Matrix(param_count, 1);
            
            double error = 0.0;
            double new_error = 0.0;
			double g_error = double.MaxValue;
			double new_g_error = 0.0;
			double v_error = 0.0;
			double new_v_error = 0.0;
            int iteration = 0;
            int convergence_count = 0;
			int overlearning_count = 0;
            double start_learning_step = 1e-3;

            do
            {
                for (int i = 0; i < layer_count; i++)
                {
                    weights[i].CopyTo(reserve_weights[i]);
                    biases[i].CopyTo(reserve_biases[i]);
                }

                if (iteration % param_count == 0)
                {
                    // find average batch gradient
					error = find_batch_gradient(training_set, avg_gradient_mat, error_weights);
                    avg_gradient_mat.CopyTo(prev_gradient_mat);
                }
                else
                {
                    avg_gradient_mat.CopyTo(prev_cunjugate_gradient);
                    // find conjugate gradient
                    error = find_batch_gradient(training_set, avg_gradient_mat, error_weights);
                    double beta = find_conjugate_beta(prev_gradient_mat, avg_gradient_mat);
                    avg_gradient_mat.CopyTo(prev_gradient_mat);
                    for (int i = 0; i < param_count; i++)
                        avg_gradient_mat[i, 0] += beta * prev_cunjugate_gradient[i, 0];

                    g_error = Math.Min(new_g_error, g_error);
                    v_error = new_v_error;
                }
                
                new_error = error;
                if (iteration == 0)
                {
					g_error = meansqr_err(generalization_set, error_weights);
					v_error = meansqr_err(validation_set, error_weights);
                    report_delegate(iteration, error, g_error, v_error);
                }

                // update parameters from gradient
                double learning_step = start_learning_step;
                bool descended = false;
                int descend_count = 0;
                do
                {
                    descend_using_gradient(reserve_weights, reserve_biases, learning_step, avg_gradient_mat);
                    // did we descend?

                    double descend_error = meansqr_err(training_set, error_weights);
                    if (descend_error < new_error)
                    {
                        // yes we did
                        new_error = descend_error;
                        learning_step *= 2.0;
                        start_learning_step *= 1.1;
                        descended = true;
                    }
                    else
                    {
                        if (learning_step == start_learning_step)
                        {
                            // our very first descend is too steep
                            start_learning_step *= 0.1;
                            if (start_learning_step < 1e-280)
                            {
                                start_learning_step = 1e-280;
                                descended = false;
                                new_error = descend_error;
                            }
                            else
                                descended = true;
                            learning_step = start_learning_step;
                        }
                        else
                        {
                            // we found minimum interval
                            new_error = golden_section_search(training_set, reserve_weights, reserve_biases, avg_gradient_mat,
                                error_weights, learning_step / 4.0, learning_step, 8);
                            descended = false;
                        }
                    }
                    descend_count++;
                } while (descended && !stop_flag && (descend_count < 50));

                if (new_error >= error)
                {
                    // return to backup
                    for (int i = 0; i < layer_count; i++)
                    {
                        reserve_weights[i].CopyTo(weights[i]);
                        reserve_biases[i].CopyTo(biases[i]);
                    }
                    new_error = error;
                }
                else
                {
					new_v_error = meansqr_err(validation_set, error_weights);
					new_g_error = meansqr_err(generalization_set, error_weights);
					if (new_g_error > g_error)
						overlearning_count++;
					else
						overlearning_count = 0;
                    report_delegate(iteration + 1, new_error, new_g_error, new_v_error);
                }
                
                iteration++;

                if (Math.Abs((new_error - error) / error) < convergence_epsilon)
                    convergence_count++;
                else
                    convergence_count = 0;
            } while ((convergence_count < convergence_count_limit && overlearning_count < overlearning_count_limit && 
                iteration < iteration_limit && error > convergence_epsilon || iteration == 1) && !stop_flag);
        }

        public void train_BP_stochastic(IList<TrainingPair> training_set, IList<TrainingPair> generalization_set,
            IList<TrainingPair> validation_set, Matrix error_weights,
            double convergence_epsilon, int iteration_limit, Action<int, double, double, double> report_delegate, ref bool stop_flag,
            int convergence_count_limit, int overlearning_count_limit, double init_speed, double spd_increase, double spd_decrease, int minibatch = 1)
        {
            // let's create required structures
            List<TrainingPair> minibatch_pairs = new List<TrainingPair>();
            Matrix[] reserve_weights = new Matrix[layer_count];
            Matrix[] reserve_biases = new Matrix[layer_count];
            Matrix[] sensitivities = new Matrix[layer_count];
            param_count = 0;
            for (int i = 0; i < layer_count; i++)
            {
                param_count += weights[i].rows * weights[i].cols;
                param_count += biases[i].rows;
                reserve_weights[i] = weights[i].Copy();
                reserve_biases[i] = biases[i].Copy();
                sensitivities[i] = new Matrix(biases[i].rows, 1);
            }
            Matrix gradient = new Matrix(param_count, 1);

            double error = 0.0;
            double new_error = 0.0;
            double t_error = 0.0;
            double g_error = double.MaxValue;
            double new_g_error = 0.0;
            double v_error = 0.0;
            double new_v_error = 0.0;
            int iteration = 0;
            int epoch = 0;
            int convergence_count = 0;
            int overlearning_count = 0;
            int chosen_point = 0;
            double learning_rate = init_speed;
            double lr_scale = 1.0 / Math.Sqrt(param_count);

            Random rng = new Random();

            do
            {
                chosen_point = rng.Next() % training_set.Count;
                TrainingPair pair = training_set[chosen_point];
                if (minibatch > 1)
                {
                    minibatch_pairs.Clear();
                    for (int i = 0; i < minibatch; i++)
                        minibatch_pairs.Add(training_set[(chosen_point + i) % training_set.Count]);
                }

                for (int i = 0; i < layer_count; i++)
                {
                    weights[i].CopyTo(reserve_weights[i]);
                    biases[i].CopyTo(reserve_biases[i]);
                }

                if (iteration == 0)
                {
                    t_error = meansqr_err(training_set, error_weights);
                    g_error = meansqr_err(generalization_set, error_weights);
                    v_error = meansqr_err(validation_set, error_weights);
                    report_delegate(epoch, t_error, g_error, v_error);
                    if (minibatch == 1)
                        error = backpropagation_gradient(pair, gradient, sensitivities, error_weights);
                    else
                        error = find_batch_gradient(minibatch_pairs, gradient, error_weights);
                }
                else
                {
                    if (minibatch == 1)
                        error = backpropagation_gradient(pair, gradient, sensitivities, error_weights);
                    else
                        error = find_batch_gradient(minibatch_pairs, gradient, error_weights);
                }

                new_error = error;

                int descend_iter = 0;

                do
                {
                    // update parameters from gradient
                    descend_using_gradient(reserve_weights, reserve_biases, lr_scale * learning_rate, gradient);
                    // did we descend?
                    double descend_error = (minibatch == 1) ? sqr_err(pair, error_weights) : meansqr_err(minibatch_pairs, error_weights);
                    if (descend_error < new_error)
                    {
                        // yes we did
                        new_error = descend_error;
                        learning_rate *= spd_increase;
                        descend_iter = 50;
                    }
                    else
                    {
                        learning_rate = Math.Max(learning_rate * spd_decrease, 1e-200);
                    }
                    descend_iter++;
                } while (descend_iter < 30 && (spd_decrease != 1.0));

                if (new_error >= error)
                {
                    // return to backup
                    for (int i = 0; i < layer_count; i++)
                    {
                        reserve_weights[i].CopyTo(weights[i]);
                        reserve_biases[i].CopyTo(biases[i]);
                    }
                    new_error = error;
                }

                iteration += minibatch;
                if (iteration > training_set.Count)
                {
                    iteration = 0;
                    t_error = meansqr_err(training_set, error_weights);
                    new_v_error = meansqr_err(validation_set, error_weights);
                    new_g_error = meansqr_err(generalization_set, error_weights);
                    if (new_g_error > g_error)
                        overlearning_count++;
                    else
                        overlearning_count = 0;
                    report_delegate(epoch + 1, t_error, new_g_error, new_v_error);

                    if (Math.Abs((new_error - error) / error) < convergence_epsilon)
                        convergence_count++;
                    else
                        convergence_count = 0;

                    g_error = Math.Min(new_g_error, g_error);
                    v_error = new_v_error;

                    epoch++;
                }                
            } while ((convergence_count < convergence_count_limit && overlearning_count < overlearning_count_limit &&
                epoch < iteration_limit && error > convergence_epsilon || epoch == 0) && !stop_flag);
        }

        public void train_BP_stochastic_process(IList<TrainingProcess> training_set, IList<TrainingProcess> generalization_set,
            IList<TrainingProcess> validation_set, TansigAnn emulator,
            double convergence_epsilon, int iteration_limit, Action<int, double, double, double> report_delegate, ref bool stop_flag,
            int convergence_count_limit, int overlearning_count_limit, double init_speed, double spd_increase, double spd_decrease, int minibatch = 1)
        {
            // let's create required structures
            List<TrainingProcess> minibatch_pairs = new List<TrainingProcess>();
            Matrix[] reserve_weights = new Matrix[layer_count];
            Matrix[] reserve_biases = new Matrix[layer_count];
            Matrix[] sensitivities = new Matrix[layer_count];
            param_count = 0;
            for (int i = 0; i < layer_count; i++)
            {
                param_count += weights[i].rows * weights[i].cols;
                param_count += biases[i].rows;
                reserve_weights[i] = weights[i].Copy();
                reserve_biases[i] = biases[i].Copy();
                sensitivities[i] = new Matrix(biases[i].rows, 1);
            }
            Matrix gradient = new Matrix(param_count, 1);

            double error = 0.0;
            double new_error = 0.0;
            double t_error = 0.0;
            double g_error = double.MaxValue;
            double new_g_error = 0.0;
            double v_error = 0.0;
            double new_v_error = 0.0;
            int iteration = 0;
            int epoch = 0;
            int convergence_count = 0;
            int overlearning_count = 0;
            int chosen_point = 0;
            double learning_rate = init_speed;
            double lr_scale = 1.0 / Math.Sqrt(param_count);

            //Random rng = new Random();

            do
            {
                chosen_point = (chosen_point + 1) % training_set.Count;
                TrainingProcess process = training_set[chosen_point];
                if (minibatch > 1)
                {
                    minibatch_pairs.Clear();
                    for (int i = 0; i < minibatch; i++)
                        minibatch_pairs.Add(training_set[(chosen_point + i) % training_set.Count]);
                }

                for (int i = 0; i < layer_count; i++)
                {
                    weights[i].CopyTo(reserve_weights[i]);
                    biases[i].CopyTo(reserve_biases[i]);
                }

                if (iteration == 0)
                {
                    t_error = meansqr_err(training_set, emulator);
                    g_error = meansqr_err(generalization_set, emulator);
                    v_error = meansqr_err(validation_set, emulator);
                    report_delegate(epoch, t_error, g_error, v_error);
                    if (minibatch == 1)
                        error = find_process_gradient(process, emulator, gradient);
                    else
                        error = find_process_batch_gradient(minibatch_pairs, emulator, gradient);
                }
                else
                {
                    if (minibatch == 1)
                        error = find_process_gradient(process, emulator, gradient);
                    else
                        error = find_process_batch_gradient(minibatch_pairs, emulator, gradient);
                }

                new_error = error;

                int descend_iter = 0;

                do
                {
                    // update parameters from gradient
                    descend_using_gradient(reserve_weights, reserve_biases, lr_scale * learning_rate, gradient);
                    // did we descend?
                    double descend_error = (minibatch == 1) ? 
                        process.sqrerr(emulator, this, process.init_state, process.length) : 
                        meansqr_err(minibatch_pairs, emulator);
                    if (descend_error < new_error)
                    {
                        // yes we did
                        new_error = descend_error;
                        learning_rate *= spd_increase;
                        descend_iter = 50;
                    }
                    else
                    {
                        learning_rate = Math.Max(learning_rate * spd_decrease, 1e-200);
                    }
                    descend_iter++;
                } while (descend_iter < 30 && (spd_decrease != 1.0));

                if (new_error >= error)
                {
                    // return to backup
                    for (int i = 0; i < layer_count; i++)
                    {
                        reserve_weights[i].CopyTo(weights[i]);
                        reserve_biases[i].CopyTo(biases[i]);
                    }
                    new_error = error;
                }

                iteration += minibatch;
                if (iteration > training_set.Count)
                {
                    iteration = 0;
                    t_error = meansqr_err(training_set, emulator);
                    new_v_error = meansqr_err(validation_set, emulator);
                    new_g_error = meansqr_err(generalization_set, emulator);
                    if (new_g_error > g_error)
                        overlearning_count++;
                    else
                        overlearning_count = 0;
                    report_delegate(epoch + 1, t_error, new_g_error, new_v_error);

                    if (Math.Abs((new_error - error) / error) < convergence_epsilon)
                        convergence_count++;
                    else
                        convergence_count = 0;

                    g_error = Math.Min(new_g_error, g_error);
                    v_error = new_v_error;

                    epoch++;
                }
            } while ((convergence_count < convergence_count_limit && overlearning_count < overlearning_count_limit &&
                epoch < iteration_limit && error > convergence_epsilon || epoch == 0) && !stop_flag);
        }

        public void train_CGBP_process(IList<TrainingProcess> training_set, IList<TrainingProcess> generalization_set,
            IList<TrainingProcess> validation_set, TansigAnn emulator,
            double convergence_epsilon, int iteration_limit, Action<int, double, double, double> report_delegate, ref bool stop_flag,
            int convergence_count_limit, int overlearning_count_limit)
        {
            // let's create required structures
            Matrix[] reserve_weights = new Matrix[layer_count];
            Matrix[] reserve_biases = new Matrix[layer_count];
            param_count = 0;
            for (int i = 0; i < layer_count; i++)
            {
                param_count += weights[i].rows * weights[i].cols;
                param_count += biases[i].rows;
                reserve_weights[i] = weights[i].Copy();
                reserve_biases[i] = biases[i].Copy();
            }
            Matrix avg_gradient_mat = new Matrix(param_count, 1);
            Matrix avg_gradient_proc_mat = new Matrix(param_count, 1);
            Matrix prev_gradient_mat = new Matrix(param_count, 1);
            Matrix prev_cunjugate_gradient = new Matrix(param_count, 1);

            double error = 0.0;
            double new_error = 0.0;
            double g_error = double.MaxValue;
            double new_g_error = 0.0;
            double v_error = 0.0;
            double new_v_error = 0.0;
            int iteration = 0;
            int convergence_count = 0;
            int overlearning_count = 0;
            double start_learning_step = 1e-3;

            do
            {
                for (int i = 0; i < layer_count; i++)
                {
                    weights[i].CopyTo(reserve_weights[i]);
                    biases[i].CopyTo(reserve_biases[i]);
                }

                if (iteration % param_count == 0)
                {
                    // find average batch gradient
                    error = find_process_batch_gradient(training_set, emulator, avg_gradient_mat);
                    avg_gradient_mat.CopyTo(prev_gradient_mat);
                }
                else
                {
                    avg_gradient_mat.CopyTo(prev_cunjugate_gradient);
                    // find conjugate gradient
                    error = find_process_batch_gradient(training_set, emulator, avg_gradient_mat);
                    double beta = find_conjugate_beta(prev_gradient_mat, avg_gradient_mat);
                    avg_gradient_mat.CopyTo(prev_gradient_mat);
                    for (int i = 0; i < param_count; i++)
                        avg_gradient_mat[i, 0] += beta * prev_cunjugate_gradient[i, 0];

                    g_error = Math.Min(new_g_error, g_error);
                    v_error = new_v_error;
                }

                new_error = error;
                if (iteration == 0)
                {
                    g_error = meansqr_err(generalization_set, emulator);
                    v_error = meansqr_err(validation_set, emulator);
                    report_delegate(iteration, error, g_error, v_error);
                }

                // update parameters from gradient
                double learning_step = start_learning_step;
                bool descended = false;
                int descend_count = 0;
                do
                {
                    descend_using_gradient(reserve_weights, reserve_biases, learning_step, avg_gradient_mat);
                    // did we descend?

                    double descend_error = meansqr_err(training_set, emulator);
                    if (descend_error < new_error)
                    {
                        // yes we did
                        new_error = descend_error;
                        learning_step *= 2.0;
                        start_learning_step *= 1.1;
                        descended = true;
                    }
                    else
                    {
                        if (learning_step == start_learning_step)
                        {
                            // our very first descend is too steep
                            start_learning_step *= 0.1;
                            if (start_learning_step < 1e-280)
                            {
                                start_learning_step = 1e-280;
                                descended = false;
                                new_error = descend_error;
                            }
                            else
                                descended = true;
                            learning_step = start_learning_step;
                        }
                        else
                        {
                            // we found minimum interval
                            double desc_err = golden_section_search(training_set, reserve_weights, reserve_biases, emulator,
                                avg_gradient_mat, learning_step / 4.0, learning_step, 8);
                            new_error = desc_err;
                            descended = false;
                        }
                    }
                    descend_count++;
                } while (descended && !stop_flag && (descend_count < 200));

                if (new_error >= error)
                {
                    // return to backup
                    for (int i = 0; i < layer_count; i++)
                    {
                        reserve_weights[i].CopyTo(weights[i]);
                        reserve_biases[i].CopyTo(biases[i]);
                    }
                    if (start_learning_step <= 1e-280)
                        return;
                    new_error = error;
                }
                else
                {
                    new_v_error = meansqr_err(validation_set, emulator);
                    new_g_error = meansqr_err(generalization_set, emulator);
                    if (new_g_error > g_error)
                        overlearning_count++;
                    else
                        overlearning_count = 0;
                    report_delegate(iteration + 1, new_error, new_g_error, new_v_error);
                }

                iteration++;

                if (Math.Abs((new_error - error) / error) < convergence_epsilon)
                    convergence_count++;
                else
                    convergence_count = 0;
            } while ((convergence_count < convergence_count_limit && overlearning_count < overlearning_count_limit &&
                iteration < iteration_limit && error > convergence_epsilon || iteration == 1) && !stop_flag);
        }

        public double sqr_err(TrainingPair pair, Matrix error_weights)
        {
            Matrix output = eval(pair.input);
            double result = 0.0;
            for (int i = 0; i < output.rows; i++)
            {
                double error = pair.correct_output[i, 0] - output[i, 0];
                if (error_weights != null)
                    error *= error_weights[i, 0];
                result += error * error;
            }
            return result;
        }

        //public double sqr_err(TrainingProcess proc, Matrix error_weights)
        //{
        //    Matrix input = proc.input.Copy();
        //    double result = 0.0;
        //    for (int j = 0; j < proc.process.Length; j++)
        //    {
        //        Matrix output = eval(input);
        //        for (int i = 0; i < output.rows; i++)
        //        {
        //            double error = proc.process[j][i, 0] - output[i, 0];
        //            if (error_weights != null)
        //                error *= error_weights[i, 0];
        //            result += error * error;
        //        }
        //        // proceed on timeline
        //        output.CopyTo(input);
        //        // account for csystems
        //        int rowc = input.rows;
        //        for (int k = 0; k < proc.csystem_icount; k++)
        //        {
        //            int row = rowc - k - 1;
        //            input[row, 0] = proc.process[j][row, 0];
        //        }
        //    }
        //    return result / proc.process.Length;
        //}

        public double meansqr_err(IList<TrainingPair> pairs, Matrix error_weights)
        {
            double error = 0.0;

            object mutex = new object();

            Parallel.For(0, pairs.Count, (int i) =>
            {
                TansigAnn local_ann = new TansigAnn(weights, biases, inputs_count, layer_count, purelin_output);
                double local_error = local_ann.sqr_err(pairs[i], error_weights);
                Monitor.Enter(mutex);
                error += local_error;
                Monitor.Exit(mutex);
            });
            error /= pairs.Count;
            return error;
        }

        public double meansqr_err(IList<TrainingProcess> processes, TansigAnn emulator)
        {
            double error = 0.0;

            object mutex = new object();

            Parallel.For(0, processes.Count, (int i) =>
            {
                TansigAnn local_regulator = new TansigAnn(weights, biases, inputs_count, layer_count, purelin_output);
                TansigAnn local_emulator = new TansigAnn(emulator.weights, emulator.biases, emulator.inputs_count,
                    emulator.layer_count, emulator.purelin_output);
                TrainingProcess process = processes[i];
                double local_error = process.sqrerr(local_emulator, local_regulator, process.init_state, process.length);
                Monitor.Enter(mutex);
                error += local_error;
                Monitor.Exit(mutex);
            });
            error /= processes.Count;
            return error;
        }

        void params_to_matrix(Matrix output)
        {
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
                for (int r = 0; r < weights[l].rows; r++)
                {
                    for (int col = 0; col < weights[l].cols; col++)
                        output[gi++, 0] = weights[l][r, col];
                    output[gi++, 0] = biases[l][r, 0];
                }
        }

        void matrix_to_params(Matrix input)
        {
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
                for (int r = 0; r < weights[l].rows; r++)
                {
                    for (int col = 0; col < weights[l].cols; col++)
                        weights[l][r, col] = input[gi++, 0];
                    biases[l][r, 0] = input[gi++, 0];
                }
        }

        void add_to_params(Matrix delta)
        {
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
                for (int r = 0; r < weights[l].rows; r++)
                {
                    for (int col = 0; col < weights[l].cols; col++)
                        weights[l][r, col] += delta[gi++, 0];
                    biases[l][r, 0] += delta[gi++, 0];
                }
        }

        void descend_using_gradient(Matrix[] base_weigths, Matrix[] base_biases, double k, Matrix gradient)
        {
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
                for (int r = 0; r < weights[l].rows; r++)
                {
                    for (int col = 0; col < weights[l].cols; col++)
                        weights[l][r, col] = base_weigths[l][r, col] - k * gradient[gi++, 0];
                    biases[l][r, 0] = base_biases[l][r, 0] - k * gradient[gi++, 0];
                }
        }

        void descend_using_gradient(Matrix[] base_weigths, Matrix[] base_biases, Matrix learning_rates, Matrix direction)
        {
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
                for (int r = 0; r < weights[l].rows; r++)
                {
                    for (int col = 0; col < weights[l].cols; col++)
                        weights[l][r, col] = base_weigths[l][r, col] - learning_rates[gi, 0] * direction[gi++, 0];
                    biases[l][r, 0] = base_biases[l][r, 0] - learning_rates[gi, 0] * direction[gi++, 0];
                }
        }

        public void fwdpropagation_process(Matrix input, Matrix output, Matrix[] jacobians)
        {
            Matrix local_output = eval(input);
            if (output != null)
            {
                for (int i = 0; i < local_output.rows; i++)
                    output[i, 0] = local_output[i, 0];
            }

            // first matrix in jacobians is derivatives of input by parameters
            // last one is the one to be the derivative of output of neural network
            // derivatives in the middle are partial derivatives of layer net outputs by parameters
            // derivative matrix column count = parameter count

            int gi = 0;
            for (int l = 0; l < layer_count; l++)                   // cycle layers
            {
                Matrix deriv_prev = jacobians[l];
                Matrix deriv = jacobians[l + 1];
                Matrix w = weights[l];
                Matrix b = biases[l];
                Matrix x = l == 0 ? input : outputs[l - 1];

                // let's propagate derivatives forward
                for (int n = 0; n < outputs[l].rows; n++)           // cycle neurons
                {
                    for (int i = 0; i < w.cols; i++)                // cycle inputs
                    {
                        // cycle all derivatives from previous deriv
                        for (int param = 0; param < deriv_prev.cols; param++)
                        {
                            if (i == 0)
                                deriv[n, param] = w[n, i] * deriv_prev[i, param];
                            else
                                if (i < deriv_prev.rows)
                                    deriv[n, param] += w[n, i] * deriv_prev[i, param];
                        }
                        // and now local derivative       
                        deriv[n, gi++] += x[i, 0];
                    }
                    deriv[n, gi++] += 1.0;
                }

                // if we're in tansig layer, let's apply it's derivative
                if (l != layer_count - 1 || !purelin_output)
                {
                    for (int n = 0; n < outputs[l].rows; n++)       // cycle outputs
                    {
                        double net = net_inputs[l][n, 0];
                        double d = tanh_factor * tanh_deriv(net_factor * net) * net_factor;
                        // chain rule for layer output 
                        for (int param = 0; param < deriv.cols; param++)
                        {
                            deriv[n, param] = d * deriv[n, param];
                        }
                    }
                }
            }
        }

        public void fwdpropagation_derivs(Matrix input, Matrix output, Matrix[] jacobians)
        {
            Matrix local_output = eval(input);
            if (output != null)
            {
                for (int i = 0; i < local_output.rows; i++)
                    output[i, 0] = local_output[i, 0];
            }

            for (int l = 0; l < layer_count; l++)                   // cycle layers
            {
                Matrix deriv_prev = jacobians[l];
                Matrix deriv = jacobians[l + 1];
                Matrix w = weights[l];
                Matrix b = biases[l];
                Matrix x = l == 0 ? input : outputs[l - 1];

                // let's propagate derivatives forward
                Matrix.Multiply(w, deriv_prev, ref deriv);

                // if we're in tansig layer, let's apply it's derivative
                if (l != layer_count - 1 || !purelin_output)
                {
                    for (int n = 0; n < outputs[l].rows; n++)       // cycle outputs
                    {
                        double net = net_inputs[l][n, 0];
                        double d = tanh_factor * tanh_deriv(net_factor * net) * net_factor;
                        // chain rule for layer output 
                        for (int param = 0; param < deriv.cols; param++)
                        {
                            deriv[n, param] = d * deriv[n, param];
                        }
                    }
                }
            }
        }

        double backpropagation_gradient(TrainingPair pair, Matrix gradient, Matrix[] sensitivities, Matrix error_weights)
        {
            double msqr_error = 0.0;
            eval(pair.input);
            // output layer sensitivities
            Matrix s = sensitivities[layer_count - 1];
            for (int r = 0; r < outputs[layer_count - 1].rows; r++)
            {
                double error = pair.correct_output[r, 0] - outputs[layer_count - 1][r, 0];
                if (error_weights != null)
                    error *= error_weights[r, 0];
                s[r, 0] = -2.0 * error;
                if (!purelin_output)
                    s[r, 0] *= tanh_factor * tanh_deriv(net_factor * net_inputs[layer_count - 1][r, 0]) * net_factor;
                if (error_weights != null)
                    s[r, 0] *= error_weights[r, 0];
                msqr_error += error * error;
            }
            // hidden layers sensetivities
            for (int l = layer_count - 2; l >= 0; l--)
            {
                s = sensitivities[l];
                Matrix.Multiply(weights[l + 1], sensitivities[l + 1], ref s, true);
                for (int r = 0; r < outputs[l].rows; r++)
                    s[r, 0] *= tanh_factor * tanh_deriv(net_factor * net_inputs[l][r, 0]) * net_factor;
            }
            // now let's form a gradient column-vector from sensitivities
            int gi = 0;
            for (int l = 0; l < layer_count; l++)
            {
                // l-layer gradient
                for (int r = 0; r < weights[l].rows; r++)
                {
                    double sens = sensitivities[l][r, 0];
                    // weights
                    for (int c = 0; c < weights[l].cols; c++)
                    {
                        double input = l == 0 ? pair.input[c, 0] : outputs[l - 1][c, 0];
                        gradient[gi++, 0] = sens * input;
                    }
                    // biases
                    gradient[gi++, 0] = sens;
                }
            }
            // return mean square error for this training pair
            return msqr_error;
        }

        double find_batch_gradient(IList<TrainingPair> batch, Matrix gradient, Matrix error_weights)
        {
            double error = 0.0;
            gradient.Fill(0.0);
            
            object mutex = new object();

            Parallel.For(0, batch.Count, (int i) =>
                {
                    TansigAnn local_ann = new TansigAnn(weights, biases, inputs_count, layer_count, purelin_output);
                    Matrix local_gradient = new Matrix(param_count, 1);
                    Matrix[] local_sensitivities = new Matrix[layer_count];
                    for (int j = 0; j < layer_count; j++)
                        local_sensitivities[j] = new Matrix(weights[j].rows, 1);
                    double local_error = local_ann.backpropagation_gradient(batch[i], local_gradient, 
                        local_sensitivities, error_weights);
                    Monitor.Enter(gradient);
                    error += local_error;
                    Matrix.Add(local_gradient, gradient, ref gradient);
                    Monitor.Exit(gradient);
                });

            error /= batch.Count;
            for (int p = 0; p < gradient.rows; p++)
                gradient[p, 0] /= batch.Count;
            return error;
        }

        double find_process_gradient(TrainingProcess process, TansigAnn emulator, Matrix gradient)
        {
            double error = 0.0;
            gradient.Fill(0.0);

            Matrix input_state = process.init_state.Copy();
            Matrix output_state = new Matrix(input_state.rows, 1);

            // якобианы выходов слоёв регулятора от параметров регулятора
            Matrix[] control_jacobians = new Matrix[layer_count + 1];
            control_jacobians[0] = new Matrix(inputs_count, param_count);
            for (int j = 0; j < layer_count; j++)
                control_jacobians[j + 1] = new Matrix(weights[j].rows, param_count);

            // якобианы выходов слоёв эмулятора от параметров регулятора
            Matrix[] emulator_jacobians = new Matrix[emulator.layer_count + 1];
            emulator_jacobians[0] = new Matrix(emulator.inputs_count, param_count);
            for (int j = 1; j <= emulator.layer_count; j++)
                emulator_jacobians[j] = new Matrix(emulator.weights[j - 1].rows, param_count);

            Matrix global_jacobian = new Matrix(inputs_count, param_count);

            double finish_error = 0.0;
            double process_error = 0.0;     // ошибка на конкретном процессе
            bool finished = false;
            int iter = 0;                   // ограничим число итераций

            while (!finished && iter < 1000)
            {
                double new_error = process.step(emulator, this, input_state, output_state, emulator_jacobians,
                    control_jacobians, global_jacobian, gradient, out finished, iter);
                if (finished)
                    finish_error = new_error;
                else
                    process_error += new_error;
                // шаг во времени
                output_state.CopyTo(input_state);
                iter++;
            }
            process.length = iter;
            if (iter > 1)
                process_error /= iter - 1.0;
            error += process_error + finish_error;

            return error;
        }

        double find_process_batch_gradient(IList<TrainingProcess> batch, TansigAnn emulator, Matrix gradient)
        {
            double error = 0.0;
            gradient.Fill(0.0);

            object mutex = new object();

            Parallel.For(0, batch.Count, (int i) =>
            {
                // локальные копии сетей регулятора и эмулятора
                TansigAnn local_ann = new TansigAnn(weights, biases, inputs_count, layer_count, purelin_output);
                TansigAnn local_emulator = new TansigAnn(emulator.weights, emulator.biases, emulator.inputs_count, 
                    emulator.layer_count, emulator.purelin_output);

                Matrix local_gradient = new Matrix(param_count, 1);
                Matrix input_state = batch[i].init_state.Copy();
                Matrix output_state = new Matrix(input_state.rows, 1);
                
                // якобианы выходов слоёв регулятора от параметров регулятора
                Matrix[] control_jacobians = new Matrix[layer_count + 1];
                control_jacobians[0] = new Matrix(inputs_count, param_count);
                for (int j = 0; j < layer_count; j++)
                    control_jacobians[j + 1] = new Matrix(weights[j].rows, param_count);

                // якобианы выходов слоёв эмулятора от параметров регулятора
                Matrix[] emulator_jacobians = new Matrix[emulator.layer_count + 1];
                emulator_jacobians[0] = new Matrix(emulator.inputs_count, param_count);
                for (int j = 1; j <= emulator.layer_count; j++)
                    emulator_jacobians[j] = new Matrix(emulator.weights[j - 1].rows, param_count);

                Matrix global_jacobian = new Matrix(inputs_count, param_count);

                double finish_error = 0.0;
                double process_error = 0.0;     // ошибка на конкретном процессе
                bool finished = false;
                TrainingProcess process = batch[i];
                int iter = 0;                   // ограничим число итераций

                while (!finished && iter < 1000)
                {
                    double new_error = process.step(emulator, this, input_state, output_state, emulator_jacobians, 
                        control_jacobians, global_jacobian, local_gradient, out finished, iter);
                    if (finished)
                        finish_error = new_error;
                    else
                        process_error += new_error;
                    // шаг во времени
                    output_state.CopyTo(input_state);
                    iter++;
                }
                process.length = iter;
                if (iter > 1)
                    process_error /= iter - 1.0;
                Monitor.Enter(gradient);
                error += process_error + finish_error;
                Matrix.Add(local_gradient, gradient, ref gradient);
                Monitor.Exit(gradient);
            });

            error /= batch.Count;
            for (int p = 0; p < gradient.rows; p++)
                gradient[p, 0] /= batch.Count;
            return error;
        }

        double golden_section_search(IList<TrainingPair> batch, Matrix[] base_weights, Matrix[] base_biases, Matrix gradient,
            Matrix error_weights, double a, double b, int iter_limit)
        {
            // we didn't, let's find minimum point now using golden section search
            double result = 0.0;
            double tau = 0.618;
            double c = a + (1.0 - tau) * (b - a);
            double d = b - (1.0 - tau) * (b - a);
            double Fc = 0.0;
            double Fd = 0.0;
            int prev_search = 0;
            for (int i = 0; i < iter_limit; i++)
            {
                // apply parameters in point c
                if (prev_search < 1)
                {
                    descend_using_gradient(base_weights, base_biases, c, gradient);
                    Fc = meansqr_err(batch, error_weights);
                }
                // apply parameters in point d
                if (prev_search > -1)
                {
                    descend_using_gradient(base_weights, base_biases, d, gradient);
                    Fd = meansqr_err(batch, error_weights);
                }
                // judge and change
                if (Fc < Fd)
                {
                    result = Fc;
                    b = d;
                    d = c;
                    c = a + (1.0 - tau) * (b - a);
                    Fd = Fc;
                    prev_search = -1;                    
                }
                else
                {
                    result = Fd;
                    a = c;
                    c = d;
                    d = b - (1.0 - tau) * (b - a);
                    Fc = Fd;
                    prev_search = 1;
                }
            }
            return result;
        }

        double golden_section_search(IList<TrainingProcess> batch, Matrix[] base_weights, Matrix[] base_biases,
            TansigAnn emulator, Matrix gradient, double a, double b, int iter_limit)
        {
            // we didn't, let's find minimum point now using golden section search
            double result = 0.0;
            double tau = 0.618;
            double c = a + (1.0 - tau) * (b - a);
            double d = b - (1.0 - tau) * (b - a);
            double Fc = 0.0;
            double Fd = 0.0;
            int prev_search = 0;
            for (int i = 0; i < iter_limit; i++)
            {
                // apply parameters in point c
                if (prev_search < 1)
                {
                    descend_using_gradient(base_weights, base_biases, c, gradient);
                    Fc = meansqr_err(batch, emulator);
                }
                // apply parameters in point d
                if (prev_search > -1)
                {
                    descend_using_gradient(base_weights, base_biases, d, gradient);
                    Fd = meansqr_err(batch, emulator);
                }
                // judge and change
                if (Fc < Fd)
                {
                    result = Fc;
                    b = d;
                    d = c;
                    c = a + (1.0 - tau) * (b - a);
                    Fd = Fc;
                    prev_search = -1;
                }
                else
                {
                    result = Fd;
                    a = c;
                    c = d;
                    d = b - (1.0 - tau) * (b - a);
                    Fc = Fd;
                    prev_search = 1;
                }
            }
            return result;
        }

        double find_conjugate_beta(Matrix prev_grad, Matrix cur_grad)
        {
            double upper = 0.0;
            double lower = 0.0;
            for (int i = 0; i < prev_grad.rows; i++)
            {
                upper += cur_grad[i, 0] * cur_grad[i, 0];
                lower += prev_grad[i, 0] * prev_grad[i, 0];
            }
            return upper / lower;
        }
    }
}
