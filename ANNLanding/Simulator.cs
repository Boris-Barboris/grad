using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
//using System.Windows.Media.Media3D;
using XnaGeometry;

namespace ANNLanding
{

    public interface IControlSystem
    {
        Matrix ApplyControl(Matrix state, double time);
    }

    /// <summary>
    /// Симулятор, управляющий вычислительным экспериментом над моделью самолёта.
    /// </summary>
    public class Simulator
    {
        public readonly PlaneModel model;

        public Simulator()
        {
            model = new PlaneModel();
        }

        /// <summary>
        /// Инициализатор состояния модели для примитивной отладки в первой вкладке интерфейса.
        /// </summary>
        /// <param name="speed">Воздушная скорость</param>
        /// <param name="height">Высота</param>
        /// <param name="start_AoA">Начальный угол атаки</param>
        public void SandboxInit(double vx, double vy, double height, double x, double start_AoA, double cz)
        {
            model.position = new Vector3(x, height, 0.0);
            // не трогаем вращение, будем просто лететь по оси X стартовой системы координат
            model.rotation =
                Quaternion.CreateFromAxisAngle(Vector3.UnitZ, -start_AoA * PlaneModel.dgr2rad);// *
                //Quaternion.CreateFromAxisAngle(Vector3.UnitY, -start_Sideslip * PlaneModel.dgr2rad);// *
                //Quaternion.CreateFromAxisAngle(Vector3.UnitY, 0.5 * Math.PI);

            model.rotation.Normalize();
            model.velocity = new Vector3(vx, vy, 0.0);
            model.ang_vel = new Vector3();
            model.control_surface = new Vector3(0.0, 0.0, cz);
            model.initialize();
            fill_pure_state(emul_state_pure, model);
        }

        public static void fill_pure_state(Matrix state, PlaneModel model)
        {
            state[0, 0] = model.position.X;
            state[1, 0] = model.position.Y;
            state[2, 0] = model.velocity.X;
            state[3, 0] = model.velocity.Y;
            state[4, 0] = model.ang_vel.Z;
            state[5, 0] = model.AoA;
            state[6, 0] = model.control_surface.Z;
        }

        Matrix emul_state_pure = new Matrix(8, 1);

        /// <summary>
        /// Инициализатор состояния модели из вектора состояния в "чистом" масштабе (ненормированном)
        /// </summary>
        /// <param name="state">Столбец-состояние</param>
        public void StateInit(Matrix state)
        {
            model.position = new Vector3(state[0, 0], state[1, 0], 0.0);
            model.velocity = new Vector3(state[2, 0], state[3, 0], 0.0);
            model.ang_vel = new Vector3(0.0, 0.0, state[4, 0]);
            model.init_rotation_angles(state[5, 0], 0.0, 0.0);
            model.control_surface = new Vector3(0.0, 0.0, state[6, 0]);
            model.initialize();
            // return corrected angles
            state[5, 0] = model.AoA;
        }

        public enum FlightData
        {
            AoA,
//            Sideslip,
//            Bank,
//            ang_vel_x,
//            ang_vel_y,
            ang_vel_z,
            vel_x,
            vel_y,
//            vel_z,
//            csurf_x,
//            csurf_y,
            csurf_z,
            signal,
            Speed,
            Altitude,
            X,
//            Z,
            time
        }

        /// <summary>
        /// Результат эксперимента
        /// </summary>
        public Dictionary<FlightData, List<double>> experiment_result = new Dictionary<FlightData, List<double>>();

        /// <summary>
        /// Результат эксперимента с сетью-эмулятором
        /// </summary>
        public Dictionary<FlightData, List<double>> emulator_result = new Dictionary<FlightData, List<double>>();

        public delegate bool StateDelegate(Matrix state, double time, PlaneModel model);

        /// <summary>
        /// Главная функция, проводящая эксперимент.
        /// </summary>
        /// <param name="csystem">Система управления</param>
        /// <param name="callback">Служебная функция</param>
        /// <param name="tmax">Время моделирования</param>
        /// <param name="deltaT">Шаг интегрирования</param>
        public void Simulate(IControlSystem csystem, StateDelegate callback, double tmax, double deltaT = 0.02)
        {
            experiment_result.Clear();
            foreach (var enum_val in typeof(FlightData).GetEnumValues())
            {
                // аллокация контейнеров для истории полёта
                experiment_result[(FlightData)enum_val] = new List<double>((int)Math.Ceiling(tmax / deltaT) + 1);
            }

            double time = 0.0;
            Matrix pure_state = new Matrix(8, 1);
            fill_pure_state(pure_state, model);

            if (callback != null)
                callback(pure_state, time, model);
            report_states(time, 0.0);

            // главный цикл
            while (time <= tmax)
            {
                Vector3 inputs = model.control_surface;
                Matrix cinputs = null;
                if (csystem != null)
                {
                    cinputs = csystem.ApplyControl(get_state(), time);
                    inputs.X = cinputs[0, 0];
                    inputs.Y = cinputs[1, 0];
                    inputs.Z = cinputs[2, 0];
                }
                pure_state[7, 0] = inputs.Z;

                model.integrate(deltaT, inputs);
                time += deltaT;

                if (callback != null)
                {
                    fill_pure_state(pure_state, model);
                    if (callback(pure_state, time, model))
                        return;
                }
                report_states(time, inputs.Z);
            }
        }

        /// <summary>
        /// Главная функция, проводящая эксперимент над эмулятором модели.
        /// </summary>
        /// <param name="csystem">Система управления</param>
        /// <param name="callback">Служебная функция</param>
        /// <param name="tmax">Время моделирования</param>
        /// <param name="deltaT">Шаг интегрирования</param>
        public void SimulateAnn(TansigAnn emulator, IControlSystem csystem, StateDelegate callback, double tmax, double deltaT = 0.02)
        {
            emulator_result.Clear();
            foreach (var enum_val in typeof(FlightData).GetEnumValues())
            {
                // аллокация контейнеров для истории полёта
                emulator_result[(FlightData)enum_val] = new List<double>((int)Math.Ceiling(tmax / deltaT) + 1);
            }

            double time = 0.0;

            if (callback != null)
                callback(emul_state_pure, time, model);
            report_states_emulator(time, 0.0);

            Matrix input_mat = new Matrix(6, 1);

            // главный цикл
            while (time <= tmax)
            {
                if (csystem != null)
                {
                    Matrix cinputs = csystem.ApplyControl(emul_state_pure, time);
                    emul_state_pure[7, 0] = cinputs[2, 0];
                }
                EmulatorTrainer.fill_norm_input_matrix(input_mat, emul_state_pure);
                input_mat[5, 0] = Rescale.norm_input(emul_state_pure[7, 0], 5);

                Matrix state_delta = emulator.eval(input_mat);
                
                // особые состояния (актуатор)
                Vector3 csurf = PlaneModel.update_control(new Vector3(0.0, 0.0, emul_state_pure[7, 0]),
                    new Vector3(0.0, 0.0, emul_state_pure[6, 0]), deltaT);
                for (int i = 0; i < 6; i++)
                    emul_state_pure[i, 0] += Rescale.pure_output(state_delta[i, 0], i);
                emul_state_pure[6, 0] = csurf.Z;

                time += deltaT;

                report_states_emulator(time, emul_state_pure[7, 0]);
                if (callback != null)
                    if (callback(emul_state_pure, time, model))
                        return;
            }
        }

        void report_states(double time, double signal)
        {
            experiment_result[FlightData.AoA].Add(model.AoA);
            //experiment_result[FlightData.Sideslip].Add(model.Sideslip);
            //experiment_result[FlightData.Bank].Add(model.Bank);
            //experiment_result[FlightData.ang_vel_x].Add(model.ang_vel.X);
            //experiment_result[FlightData.ang_vel_y].Add(model.ang_vel.Y);
            experiment_result[FlightData.ang_vel_z].Add(model.ang_vel.Z);
            experiment_result[FlightData.vel_x].Add(model.velocity.X);
            experiment_result[FlightData.vel_y].Add(model.velocity.Y);
            //experiment_result[FlightData.vel_z].Add(model.velocity.Z);
            //experiment_result[FlightData.csurf_x].Add(model.control_surface.X);
            //experiment_result[FlightData.csurf_y].Add(model.control_surface.Y);
            experiment_result[FlightData.csurf_z].Add(model.control_surface.Z);
            experiment_result[FlightData.signal].Add(signal);
            experiment_result[FlightData.Speed].Add(model.Speed);
            experiment_result[FlightData.Altitude].Add(model.Altitude);
            experiment_result[FlightData.X].Add(model.position.X);
            //experiment_result[FlightData.Z].Add(model.position.Z);
            experiment_result[FlightData.time].Add(time);
        }

        void report_states_emulator(double time, double signal)
        {
            emulator_result[FlightData.X].Add(emul_state_pure[0, 0]);
            emulator_result[FlightData.Altitude].Add(emul_state_pure[1, 0]);
            emulator_result[FlightData.vel_x].Add(emul_state_pure[2, 0]);
            emulator_result[FlightData.vel_y].Add(emul_state_pure[3, 0]);
            emulator_result[FlightData.ang_vel_z].Add(emul_state_pure[4, 0]);
            emulator_result[FlightData.AoA].Add(emul_state_pure[5, 0]);
            emulator_result[FlightData.csurf_z].Add(emul_state_pure[6, 0]);
            emulator_result[FlightData.signal].Add(signal);

            // speed
            double vx = emul_state_pure[2, 0];
            double vy = emul_state_pure[3, 0];
            emulator_result[FlightData.Speed].Add(Math.Sqrt(vx * vx + vy * vy));

            emulator_result[FlightData.time].Add(time);
        }

        Matrix get_state()
        {
            Matrix pure_state = new Matrix(7, 1);
            fill_pure_state(pure_state, model);
            return pure_state;
        }
    }
}
