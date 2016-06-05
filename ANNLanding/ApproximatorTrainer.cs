using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Windows.Media.Media3D;
using XnaGeometry;

namespace ANNLanding
{

    // Аппроксиматор будет работать со следующим вектором состояния:
    // X, Y, Z - координаты
    // Vx, Vy, Vz - проекции скорости самолёта на оси нормальной системы кооррдинат
    // wx, wy, wz - угловые скорости в свящанной системе координат
    // csx, csy, csz - положения управляющих поверхностей
    // alpha, beta, psi - угол атаки, скольжения, крена
    // cx, cy, cz - три входа для сигнала системы управления



    /// <summary>
    /// Функции для удобства работы с нормированными состояниями
    /// </summary>
    public static class Rescale
    {
        public static double norm(double pure, double bounds)
        {
            return pure / bounds;
        }

        public static double pure(double norm, double bounds)
        {
            return norm * bounds;
        }

        public static double[] state_bounds;

        static Rescale()
        {
            state_bounds = new double[18];
            // координаты
            state_bounds[0] = 500.0;
            state_bounds[1] = 500.0;
            state_bounds[2] = 500.0;
            // скорости
            state_bounds[3] = 300.0;
            state_bounds[4] = 300.0;
            state_bounds[5] = 300.0;
            // угловые скорости
            state_bounds[6] = 5.0;
            state_bounds[7] = 5.0;
            state_bounds[8] = 5.0;
            // положения управляющих поверхностей
            state_bounds[9] = 1.0;
            state_bounds[10] = 1.0;
            state_bounds[11] = 1.0;
            // углы
            state_bounds[12] = 0.4;
            state_bounds[13] = 0.4;
            state_bounds[14] = 2.0;
            // сигналы системы управления
            state_bounds[15] = 1.0;
            state_bounds[16] = 1.0;
            state_bounds[17] = 1.0;
        }
    }

    class ApproximatorTrainer
    {  
        /// <summary>
        /// Преобразует параметры модели и входы в вектор-столбец состояния дискретной системы для нейронной сети.
        /// </summary>
        /// <param name="model">Модель самолёта</param>
        /// <returns></returns>
        public static Matrix form_full_input(PlaneModel model, Matrix input)
        {
            Matrix state = new Matrix(18, 1);
            fill_state_matrix(state, model);
            state[15, 0] = Rescale.norm(input[0, 0], Rescale.state_bounds[15]);
            state[16, 0] = Rescale.norm(input[1, 0], Rescale.state_bounds[16]);
            state[17, 0] = Rescale.norm(input[2, 0], Rescale.state_bounds[17]);
            return state;
        }

        /// <summary>
        /// Заполнение матрицы состояния для аппроксиматора исходя из состояния модели
        /// </summary>
        public static void fill_state_matrix(Matrix m, PlaneModel model)
        {
            m[0, 0] = Rescale.norm(model.position.X, Rescale.state_bounds[0]);
            m[1, 0] = Rescale.norm(model.position.Y, Rescale.state_bounds[1]);
            m[2, 0] = Rescale.norm(model.position.Z, Rescale.state_bounds[2]);

            m[3, 0] = Rescale.norm(model.velocity.X, Rescale.state_bounds[3]);
            m[4, 0] = Rescale.norm(model.velocity.Y, Rescale.state_bounds[4]);
            m[5, 0] = Rescale.norm(model.velocity.Z, Rescale.state_bounds[5]);

            m[6, 0] = Rescale.norm(model.ang_vel.X, Rescale.state_bounds[6]);
            m[7, 0] = Rescale.norm(model.ang_vel.Y, Rescale.state_bounds[7]);
            m[8, 0] = Rescale.norm(model.ang_vel.Z, Rescale.state_bounds[8]);

            m[9, 0] = Rescale.norm(model.control_surface.X, Rescale.state_bounds[9]);
            m[10, 0] = Rescale.norm(model.control_surface.Y, Rescale.state_bounds[10]);
            m[11, 0] = Rescale.norm(model.control_surface.Z, Rescale.state_bounds[11]);

            m[12, 0] = Rescale.norm(model.AoA, Rescale.state_bounds[12]);
            m[13, 0] = Rescale.norm(model.Sideslip, Rescale.state_bounds[13]);
            m[14, 0] = Rescale.norm(model.Bank, Rescale.state_bounds[14]);
        }

        List<TansigAnn.TrainingPair> generated_states = new List<TansigAnn.TrainingPair>();

        Matrix pure_state = new Matrix(18, 1);

        public void generate_training_data(int set_size, double dt)
        {
            generated_states = new List<TansigAnn.TrainingPair>(set_size);
            Simulator sim = new Simulator();

            Random rnd = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < set_size; i++)
            {
                Matrix state = new Matrix(18, 1);
                for (int j = 0; j < 18; j++)
                {
                    state[j, 0] = 2.0 * (rnd.NextDouble() - 0.5);
                    pure_state[j, 0] = Rescale.pure(state[j, 0], Rescale.state_bounds[j]);
                }
                // Сформировано случайное состояние, нужно провести моделирование и выяснить эталонный вектор-выход
                sim.StateInit(pure_state);
            }
        }
    }
}
