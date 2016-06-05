using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Threading.Tasks;
//using System.Windows.Media.Media3D;
using XnaGeometry;

namespace ANNLanding
{

    /// <summary>
    /// Функции для удобства работы с нормированными состояниями
    /// </summary>
    public static class Rescale
    {
        public static double norm(double pure, double lower, double upper)
        {
            double middle = 0.5 * (lower + upper);
            return 2.0 * (pure - middle) / (upper - lower);
        }

        public static double normDelta(double pureDelta, double lower, double upper)
        {
            return 2.0 * pureDelta / (upper - lower);
        }

        public static double pureDelta(double normDelta, double lower, double upper)
        {
            return 0.5 * normDelta * (upper - lower);
        }

        public static double pure(double norm, double lower, double upper)
        {
            double middle = 0.5 * (lower + upper);
            return middle + 0.5 * norm * (upper - lower);
        }

        public static double norm_input(double pure, int projection)
        {
            return norm(pure, input_lower_bounds[projection], input_upper_bounds[projection]);
        }

        public static double pure_input(double norm, int projection)
        {
            return pure(norm, input_lower_bounds[projection], input_upper_bounds[projection]);
        }

        public static double norm_output(double pure, int projection)
        {
            return norm(pure, output_lower_bounds[projection], output_upper_bounds[projection]);
        }

        public static double pure_output(double norm, int projection)
        {
            return pure(norm, output_lower_bounds[projection], output_upper_bounds[projection]);
        }

        public static double norm_input_delta(double pure, int projection)
        {
            return normDelta(pure, input_lower_bounds[projection], input_upper_bounds[projection]);
        }

        public static double pure_input_delta(double norm, int projection)
        {
            return pureDelta(norm, input_lower_bounds[projection], input_upper_bounds[projection]);
        }

        public static double norm_output_delta(double pure, int projection)
        {
            return normDelta(pure, output_lower_bounds[projection], output_upper_bounds[projection]);
        }

        public static double pure_output_delta(double norm, int projection)
        {
            return pureDelta(norm, output_lower_bounds[projection], output_upper_bounds[projection]);
        }

        public static double[] input_lower_bounds;
        public static double[] input_upper_bounds;

        public static double[] output_lower_bounds;
        public static double[] output_upper_bounds;

        static Rescale()
        {
            RescaleInit(0.05);
        }

        /// <summary>
        /// Инициализация коэфициентов нормализации
        /// </summary>
        /// <param name="dt">Шаг времени, который будет использоваться в нейроэмуляторе.</param>
        public static void RescaleInit(double dt)
        {
            // входы в эмулятор
            input_lower_bounds = new double[6];
            input_upper_bounds = new double[6];
            // скорости
            input_lower_bounds[0] = 5.0;
            input_lower_bounds[1] = -70.0;
            input_upper_bounds[0] = 70.0;
            input_upper_bounds[1] = 50.0;
            // угловая скорость
            input_lower_bounds[2] = -1.0;
            input_upper_bounds[2] = 1.0;
            // угол атаки
            input_lower_bounds[3] = -0.4;
            input_upper_bounds[3] = 0.4;
            // положение руля
            input_lower_bounds[4] = -1.0;
            input_upper_bounds[4] = 1.0;
            // сигнал системы управления
            input_lower_bounds[5] = -1.0;
            input_upper_bounds[5] = 1.0;

            // выходы
            output_lower_bounds = new double[6];
            output_upper_bounds = new double[6];
            // шаг по координате
            output_lower_bounds[0] = input_lower_bounds[0] * dt;
            output_lower_bounds[1] = input_lower_bounds[1] * dt;
            output_upper_bounds[0] = input_upper_bounds[0] * dt;
            output_upper_bounds[1] = input_upper_bounds[1] * dt;
            // шаг по скорости (ускорение * dt)
            output_lower_bounds[2] = -30.0 * dt * 1.7;
            output_lower_bounds[3] = -30.0 * dt * 1.7;
            output_upper_bounds[2] = 30.0 * dt;
            output_upper_bounds[3] = 30.0 * dt * 1.7;
            // шаг по угловой скорости
            output_lower_bounds[4] = -7.0 * dt * 1.7;
            output_upper_bounds[4] = 7.0 * dt * 1.7;
            // шаг по углу атаки
            output_lower_bounds[5] = input_lower_bounds[3] * dt * 4.0;
            output_upper_bounds[5] = input_upper_bounds[3] * dt * 4.0 * 1.5;
        }

        public static void serialize(string filename)
        {
            using (StreamWriter writer = new StreamWriter(filename))
            {
                writer.WriteLine(string.Join(" ", input_lower_bounds.Select(v => v.ToString("G10"))));
                writer.WriteLine(string.Join(" ", input_upper_bounds.Select(v => v.ToString("G10"))));
                writer.WriteLine(string.Join(" ", output_lower_bounds.Select(v => v.ToString("G10"))));
                writer.WriteLine(string.Join(" ", output_upper_bounds.Select(v => v.ToString("G10"))));
            }
        }

        public static void deserialize(string filename)
        {
            using (StreamReader reader = new StreamReader(filename))
            {
                input_lower_bounds = reader.ReadLine().Split(' ').Select(s => double.Parse(s)).ToArray();
                input_upper_bounds = reader.ReadLine().Split(' ').Select(s => double.Parse(s)).ToArray();
                output_lower_bounds = reader.ReadLine().Split(' ').Select(s => double.Parse(s)).ToArray();
                output_upper_bounds = reader.ReadLine().Split(' ').Select(s => double.Parse(s)).ToArray();
            }
        }
    }

    class EmulatorTrainer
    {  

        /// <summary>
        /// Заполнение матрицы состояния для эмулятора исходя из состояния модели
        /// </summary>
        public static void fill_norm_input_matrix(Matrix m, PlaneModel model)
        {
            // скорости
            m[0, 0] = Rescale.norm_input(model.velocity.X, 0);
            m[1, 0] = Rescale.norm_input(model.velocity.Y, 1);
            // укловая скорость
            m[2, 0] = Rescale.norm_input(model.ang_vel.Z, 2);
            // угол атаки
            m[3, 0] = Rescale.norm_input(model.AoA, 3);
            // руль
            m[4, 0] = Rescale.norm_input(model.control_surface.Z, 4);
        }

        /// <summary>
        /// Заполнение матрицы состояния для эмулятора исходя из полной матрицы состояния
        /// </summary>
        public static void fill_norm_input_matrix(Matrix m, Matrix pure_state)
        {
            // скорости
            m[0, 0] = Rescale.norm_input(pure_state[2, 0], 0);
            m[1, 0] = Rescale.norm_input(pure_state[3, 0], 1);
            // укловая скорость
            m[2, 0] = Rescale.norm_input(pure_state[4, 0], 2);
            // угол атаки
            m[3, 0] = Rescale.norm_input(pure_state[5, 0], 3);
            // руль
            m[4, 0] = Rescale.norm_input(pure_state[6, 0], 4);
        }

        public static void fill_norm_output_matrix(Matrix m, Matrix prev_pure_state, PlaneModel model)
        {
            // координаты
            m[0, 0] = Rescale.norm_output(model.position.X - prev_pure_state[0, 0], 0);
            m[1, 0] = Rescale.norm_output(model.position.Y - prev_pure_state[1, 0], 1);
            // скорости
            m[2, 0] = Rescale.norm_output(model.velocity.X - prev_pure_state[2, 0], 2);
            m[3, 0] = Rescale.norm_output(model.velocity.Y - prev_pure_state[3, 0], 3);
            // укловая скорость
            m[4, 0] = Rescale.norm_output(model.ang_vel.Z - prev_pure_state[4, 0], 4);
            // угол атаки
            m[5, 0] = Rescale.norm_output(model.AoA - prev_pure_state[5, 0], 5);
            // актуатор
            //m[6, 0] = Rescale.norm_output(model.control_surface.Z - prev_pure_state[6, 0], 6);
        }

        List<TansigAnn.TrainingPair> training_pairs = new List<TansigAnn.TrainingPair>();
        List<TansigAnn.TrainingPair> generalization_pairs;
        List<TansigAnn.TrainingPair> validation_pairs;

        public int PairsCount
        {
            get { return training_pairs.Count + generalization_pairs.Count + validation_pairs.Count; }
        }

        public struct DummyControlSystem : IControlSystem
        {
            public Vector3 input;

            public DummyControlSystem(Vector3 signal)
            {
                input = signal;
            }

            static Matrix signal = new Matrix(3, 1);
            public Matrix ApplyControl(Matrix state, double time)
            {
                signal[0, 0] = input.X;
                signal[1, 0] = input.Y;
                signal[2, 0] = input.Z;
                return signal;
            }
        }

        public struct RngControlSystem : IControlSystem
        {
            static Random rng;

            static RngControlSystem()
            {
                rng = new Random();
            }

            static Matrix signal = new Matrix(3, 1);
            public Matrix ApplyControl(Matrix state, double time)
            {
                signal[0, 0] = 0.0;
                signal[1, 0] = 0.0;
                signal[2, 0] = 2.0 * (rng.NextDouble() - 0.5);
                return signal;
            }
        }

        public void generate_validation_points(int set_size, double dt)
        {
            validation_pairs = new List<TansigAnn.TrainingPair>(set_size);
            Simulator sim = new Simulator();

            Random rnd = new Random();
            for (int i = 0; i < set_size; i++)
            {
                Matrix pure_state = new Matrix(8, 1);
                Matrix norm_input = new Matrix(6, 1);
                Matrix norm_output = new Matrix(6, 1);
                rng_state(rnd, pure_state);
                // Сформировано случайное состояние, нужно провести моделирование и выяснить эталонный вектор-выход
                sim.StateInit(pure_state);

                // формируем вход
                for (int j = 0; j < 6; j++)
                    norm_input[j, 0] = Rescale.norm_input(pure_state[j + 2, 0], j);

                // Провести симуляцию
                sim.Simulate(new DummyControlSystem(new Vector3(0.0, 0.0, pure_state[7, 0])), null, dt / 2.0, dt);
                fill_norm_output_matrix(norm_output, pure_state, sim.model);

                TansigAnn.TrainingPair tr_pair = new TansigAnn.TrainingPair(norm_input, norm_output);
                validation_pairs.Add(tr_pair);
            }
        }

        public double[] rng_pure_lower = new double[8];
        public double[] rng_pure_upper = new double[8];

        public EmulatorTrainer()
        {
            init_rng_borders();
        }

        public void init_rng_borders()
        {
            // заполним массивы генерации
            rng_pure_lower[0] = RegulatorTrainer.Xmin;
            rng_pure_upper[0] = RegulatorTrainer.Xmax;
            rng_pure_lower[1] = RegulatorTrainer.Ymin;
            rng_pure_upper[1] = RegulatorTrainer.Ymax;

            for (int i = 2; i < 8; i++)
            {
                rng_pure_lower[i] = Rescale.input_lower_bounds[i - 2];
                rng_pure_upper[i] = Rescale.input_upper_bounds[i - 2];
            }
        }

        public bool model_out_of_safe_region(PlaneModel model)
        {
            if (Math.Abs(model.AoA) * PlaneModel.rad2dgr > 20.0)
                return true;
            if (model.Altitude < -1e-5)
                return true;
            if (model.Altitude > 20.0)
                return true;
            if (model.velocity.X < 10.0)
                return true;
            if (model.velocity.Y < -30.0)
                return true;
            if (model.velocity.X > 40.0)
                return true;
            if (model.position.X > 0.0)
                return true;
            return false;
        }

        public void rng_state(Random rnd, Matrix pure_state)
        {
            for (int j = 0; j < 8; j++)
            {
                double middle = 0.5 * (rng_pure_upper[j] + rng_pure_lower[j]);
                pure_state[j, 0] = middle + (rnd.NextDouble() - 0.5) * (rng_pure_upper[j] - rng_pure_lower[j]);
            }
        }

        public void generate_training_points(int set_size, double dt, double gen_percent = 20.0, double valid_percent = 10.0, bool valid = true)
        {
            training_pairs = new List<TansigAnn.TrainingPair>(set_size);
            Simulator sim = new Simulator();

            Random rnd = new Random();
            for (int i = 0; i < set_size; i++)
            {
                Matrix pure_state = new Matrix(8, 1);
                Matrix norm_input = new Matrix(6, 1);
                Matrix norm_output = new Matrix(6, 1);
                rng_state(rnd, pure_state);
                // Сформировано случайное состояние, нужно провести моделирование и выяснить эталонный вектор-выход
                sim.StateInit(pure_state);

                // формируем вход
                for (int j = 0; j < 6; j++)
                    norm_input[j, 0] = Rescale.norm_input(pure_state[j + 2, 0], j);

                // Провести симуляцию
                sim.Simulate(new DummyControlSystem(new Vector3(0.0, 0.0, pure_state[7, 0])), null, dt / 2.0, dt);
                fill_norm_output_matrix(norm_output, pure_state, sim.model);

                TansigAnn.TrainingPair tr_pair = new TansigAnn.TrainingPair(norm_input, norm_output);
                training_pairs.Add(tr_pair);
            }

            // Теперь разбить пары на три набора: тренировочный, генерализации и валидации
            int gen_count = (int)(set_size / 100.0 * gen_percent);
            int valid_count = (int)(set_size / 100.0 * valid_percent);

            generalization_pairs = new List<TansigAnn.TrainingPair>();

            while (gen_count > 0)
            {
                int index = rnd.Next() % training_pairs.Count;
                generalization_pairs.Add(training_pairs[index]);
                training_pairs.RemoveAt(index);
                gen_count--;
            }

            if (valid)
            {
                validation_pairs = new List<TansigAnn.TrainingPair>();
                while (valid_count > 0)
                {
                    int index = rnd.Next() % training_pairs.Count;
                    validation_pairs.Add(training_pairs[index]);
                    training_pairs.RemoveAt(index);
                    valid_count--;
                }
            }

            init_error_weights(dt);
        }

        public void update_output_normalization()
        {
            if (training_pairs.Count > 0)
            {
                double[] max_values = new double[6];
                double[] min_values = new double[6];

                for (int i = 0; i < 6; i++)
                {
                    max_values[i] = training_pairs.Max(p => p.correct_output[i, 0]);
                    min_values[i] = training_pairs.Min(p => p.correct_output[i, 0]);
                }

                // correct output_rescales
                for (int i = 0; i < 6; i++)
                {
                    double new_lower = Rescale.pure_output(min_values[i], i);
                    double new_upper = Rescale.pure_output(max_values[i], i);
                    Rescale.output_lower_bounds[i] = new_lower;
                    Rescale.output_upper_bounds[i] = new_upper;
                }
            }
        }

        Matrix error_weights = new Matrix(8, 1);

        void init_error_weights(double dt)
        {
            error_weights.Fill(1.0);
            //error_weights[4, 0] = 1.0;
            //error_weights[6, 0] = 2.0;
            //error_weights[0, 0] = 5.0;
            //error_weights[5, 0] = 1.0 / dt;
        }

        public void train_emulator_points(TansigAnn ann, Action<int, double, double, double> report_dlg, ref bool stop_flag)
        {
            ann.train_CGBP(training_pairs, generalization_pairs, validation_pairs, error_weights, 1e-10, 100000, report_dlg, ref stop_flag, 100, 50);
        }

        public void train_emulator_stoh(TansigAnn ann, Action<int, double, double, double> report_dlg, ref bool stop_flag,
            double init_rate, double rate_acc, double rate_decc)
        {
            ann.train_BP_stochastic(training_pairs, generalization_pairs, validation_pairs, error_weights, 1e-20, 1000000, report_dlg, ref stop_flag, 10000, 150, init_rate, rate_acc, rate_decc, training_pairs.Count >= 1000 ? 1 : 1);
        }

        public void train_pso_points(TansigAnn ann, Action<int, double, double, double> report_dlg, ref bool stop_flag, int p_count, double max_v,
            double inertia, double c1, double c2, double weight_span, double bias_span, bool keep_training)
        {
            ann.tain_PSO(training_pairs, generalization_pairs, validation_pairs, error_weights, p_count, 50, report_dlg,
                inertia, c1, c2, ref stop_flag, weight_span, bias_span, max_v, keep_training);
        }
    }
}
