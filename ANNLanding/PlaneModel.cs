using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using System.Windows.Media.Media3D;
using XnaGeometry;

namespace ANNLanding
{

    /// <summary>
    /// Модель планера.
    /// </summary>
    public class PlaneModel
    {
        #region Кинематические параметры

        /// <summary>
        /// Преобразование поворота, описывающее ориентацию корпуса самолёта. Соответствует повороту из связанной в 
        /// нормальную систему координат.
        /// </summary>
        public Quaternion rotation = Quaternion.Identity;
        Quaternion rotation_predict = Quaternion.Identity;

        /// <summary>
        /// Координата центра масс в стартовой системе координат.
        /// </summary>
        public Vector3 position = new Vector3();
        Vector3 position_predict = new Vector3();

        /// <summary>
        /// Высота над нулевым уровнем
        /// </summary>
        public double Altitude { get { return position.Y; } }

        /// <summary>
        /// Вектор скорости самолёта в стартовой системе координат.
        /// </summary>
        public Vector3 velocity = new Vector3();
        Vector3 velocity_predict = new Vector3();

        /// <summary>
        /// Вектор ускорения самолёта в стартовой системе координат.
        /// </summary>
        public Vector3 acceleration = new Vector3();
        Vector3 acceleration_predict = new Vector3();

        /// <summary>
        /// Угловая скорость в связанной системы координат.
        /// </summary>
        public Vector3 ang_vel = new Vector3();
        Vector3 ang_vel_predict = new Vector3();

        /// <summary>
        /// Угловое ускорение в связанной системы координат.
        /// </summary>
        public Vector3 ang_acc = new Vector3();
        Vector3 ang_acc_predict = new Vector3();

        /// <summary>
        /// Модуль скорости.
        /// </summary>
        public double Speed { get { return velocity.Length(); } }

        /// <summary>
        /// Крен.
        /// </summary>
        public double Bank
        {
            get
            {
                Vector3 Zg = Vector3.Cross(velocity, Vector3.UnitY);
                Zg.Normalize();
                Vector3 bank_v = Vector3.Cross(Zworld, Zg);
                double sign = Math.Sign(Vector3.Dot(bank_v, -velocity));
                double asin = Math.Asin(sign * bank_v.Length());
                if (Vector3.Dot(Zworld, Zg) < 0.0)
                    asin = Math.Sign(asin) * Math.PI - asin;
                return asin;
            }
        }

        #endregion

        #region Динамические параметры

        /// <summary>
        /// Плотность воздуха, принимаем константой.
        /// </summary>
        public const double air_density = 1.2466;

        /// <summary>
        /// Масса планера.
        /// </summary>
        public double mass = 230.0;

        /// <summary>
        /// Ускорение свободного падения.
        /// </summary>
        public const double g = 9.81;

        /// <summary>
        /// Скоростной напор.
        /// </summary>
        public double DynPressure { get { return 0.5 * air_density * velocity.LengthSquared(); } }
        double dyn_press = 0.0;

        /// <summary>
        /// Тензор инерции, проекции на оси связанной системы координат.
        /// </summary>
        public Vector3 MOI = new Vector3(100.0, 150.0, 110.0);

        /// <summary>
        /// Положение рулевых поверхностей.
        /// </summary>
        public Vector3 control_surface = new Vector3();

        /// <summary>
        /// Скоростной фактор рулевых поверхностей.
        /// </summary>
        public const double csurf_time_factor = 0.25;

        public const double rad2dgr = 180.0 / Math.PI;
        public const double dgr2rad = Math.PI / 180.0;

        double alpha = 0.0;
        double alpha_predict = 0.0;

        /// <summary>
        /// Угол атаки, радианы.
        /// </summary>
        public double AoA { get { return alpha; } }

        double beta = 0.0;
        double beta_predict = 0.0;

        /// <summary>
        /// Угол скольжения, радианы.
        /// </summary>
        public double Sideslip { get { return beta; } }

        #endregion

        #region Аэродинамические коэффициенты

        // Сила сопротивления Fx = - dyn_press * (Cx0 + Cxa1 * alpha + Cxa2 * alpha^2 + Cxb1 * beta + Cxb2 * beta^2)
        public double Cx0 = 0.4;
        public double Cxa1 = 0.0;
        public double Cxa2 = 30.0;
        public double Cxb1 = 0.0;
        public double Cxb2 = 13.0;

        // Подъёмная сила Fy = dyn_press * (Cy0 + Cya * alpha + Cyc * control_surface_z)
        public double Cy0 = 0.05;
        public double Cya = 39.0;
        public double Cyc = -0.05;

        // Боковая сила Fz = dyn_press * (Cz0 + Czb * beta + Cyc * control_surface_y)
        public double Cz0 = 0.0;
        public double Czb = 1.0;
        public double Czc = -0.1;

        // Момент по крену Mx = dyn_press * (Mx0 + Mxcx * control_surface_x + Mxcy * control_surface_y + Mxb * beta +... 
        // + Mxw * ang_vel_x / speed)
        public double Mx0 = 0.0;
        public double Mxcx = 0.5;
        public double Mxcy = 0.05;       // связь руля направления и крена
        public double Mxb = -0.35;        // голландский шаг
        public double Mxw = -8.0;

        // Момент по рысканью My = dyn_press * (My0 + Myb * beta + Myc * control_surface_y)
        public double My0 = 0.0;
        public double Myb = -1.0;       // стабильный самолёт
        public double Myc = 0.1;

        // Момент по тангажу Mz = dyn_press * (Mz0 + Mza * alpha + Mzc * control_surface_z)
        public double Mz0 = 0.0;
        public double Mza = -1.8;      // стабильный самолёт
        public double Mzc = 0.40;

        #endregion

        #region Методы

        /// <summary>
        /// Инициализация ориентации самолёта из трёх углов (примерная, точное решение сложно)
        /// </summary>
        /// <param name="alpha">Угол атаки</param>
        /// <param name="beta">Угол скольжения</param>
        /// <param name="bank">Крен</param>
        public void init_rotation_angles(double alpha, double beta, double bank)
        {
            Vector3 axis_local = alpha * Vector3.UnitZ + beta * Vector3.UnitY + bank * Vector3.UnitX;
            double angle = axis_local.Length();
            Quaternion local_rotation = Quaternion.CreateFromAxisAngle(axis_local, -angle);
            Vector3 Zf = Vector3.Cross(velocity, Vector3.UnitY);
            Zf.Normalize();
            Vector3 Xf = velocity;
            Xf.Normalize();
            Vector3 Yf = Vector3.Cross(Zf, Xf);
            XnaGeometry.Matrix basis_rot_m = new XnaGeometry.Matrix(
                Xf.X, Xf.Y, Xf.Z, 0.0,
                Yf.X, Yf.Y, Yf.Z, 0.0,
                Zf.X, Zf.Y, Zf.Z, 0.0,
                0.0, 0.0, 0.0, 0.0);
            Quaternion basis_rot = Quaternion.CreateFromRotationMatrix(basis_rot_m);
            basis_rot = Quaternion.Inverse(basis_rot);
            rotation = local_rotation * basis_rot;
        }

        /// <summary>
        /// Шаг численного моделирования, предиктор-корректор Эйлер-трапеций.
        /// </summary>
        /// <param name="dt">Шаг по времени.</param>
        /// <param name="inputs">Вектор входов (например, сигнал системы управления).</param>
        public void integrate(double dt, Vector3 inputs)
        {
            // Предиктор
            dyn_press = DynPressure;
            calculate_acceleration(velocity, Yworld, Zworld, dyn_press, alpha, beta, control_surface,
                out Xaworld, out Yaworld, out Zaworld, out acceleration);
            calculate_ang_acc(dyn_press, control_surface, ang_vel, alpha, beta, velocity.Length(), out ang_acc);
            euler_predict(dt);
            Vector3 csurf_p = update_control(inputs, control_surface, dt);
            
            // Корректор
            double dyn_press_p = 0.5 * air_density * velocity_predict.LengthSquared();
            update_angles(rotation_predict, velocity_predict, out Xworld_p, out Yworld_p, out Zworld_p,
                out alpha_predict, out beta_predict);
            calculate_acceleration(velocity_predict, Yworld_p, Zworld_p, dyn_press_p, alpha_predict, 
                beta_predict, csurf_p, out Xaworld_p, out Yaworld_p, out Zaworld_p, out acceleration_predict);
            calculate_ang_acc(dyn_press_p, csurf_p, ang_vel_predict, alpha_predict, beta_predict,
                velocity_predict.Length(), out ang_acc_predict);
            trapezoidal_corrector(dt);
            
            // Обновление состояний
            position = position_predict;
            velocity = velocity_predict;
            ang_vel = ang_vel_predict;
            control_surface = csurf_p;
            rotation = rotation_predict;
            update_angles(rotation, velocity, out Xworld, out Yworld, out Zworld, out alpha, out beta);            
        }

        /// <summary>
        /// Функция инициализации для обновления внутреннего состояния после задания кинематических параметров.
        /// </summary>
        public void initialize()
        {
            update_angles(rotation, velocity, out Xworld, out Yworld, out Zworld, out alpha, out beta);
        }

        public static Vector3 update_control(Vector3 inputs, Vector3 old_control, double dt)
        {
            Vector3 new_control_surface = new Vector3();
            Vector3 input_signal = clampVector(inputs, -1.0, 1.0);
            new_control_surface.X = csurf_actuator(old_control.X, input_signal.X, dt);
            new_control_surface.Y = csurf_actuator(old_control.Y, input_signal.Y, dt);
            new_control_surface.Z = csurf_actuator(old_control.Z, input_signal.Z, dt);
            return new_control_surface;
        }

        // Легко получить точное аналитическое решение
        public static double csurf_actuator(double current, double desired, double dt)
        {
            double error = desired - current;
            if (error > 0)
            {
                double time = -csurf_time_factor * Math.Log(error / csurf_time_factor);
                time += dt;
                return desired - csurf_time_factor * Math.Exp(-time / csurf_time_factor);
            }
            else if (error < 0)
            {
                double time = -csurf_time_factor * Math.Log(-error / csurf_time_factor);
                time += dt;
                return desired + csurf_time_factor * Math.Exp(-time / csurf_time_factor);
            }
            else
                return desired;
        }

        public static Vector3 clampVector(Vector3 vector, double lower, double upper)
        {
            return new Vector3(Clamp(vector.X, lower, upper),
                Clamp(vector.Y, lower, upper),
                Clamp(vector.Z, lower, upper));
        }

        public static double Clamp(double val, double lower, double upper)
        {
            return Math.Max(lower, Math.Min(val, upper));
        }

        // Оси связанной системы координат в нормальной.
        public Vector3 Xworld, Yworld, Zworld;
        Vector3 Xworld_p, Yworld_p, Zworld_p;

        /// <summary>
        /// Функция расчёта углов атаки и скольжения, а также преобразование осей связанной системы координат
        /// </summary>
        static void update_angles(Quaternion rotation, Vector3 velocity, out Vector3 Xworld, out Vector3 Yworld,
            out Vector3 Zworld, out double alpha, out double beta)
        {
            Xworld = Vector3.Transform(Vector3.UnitX, rotation);
            Yworld = Vector3.Transform(Vector3.UnitY, rotation);
            Zworld = Vector3.Transform(Vector3.UnitZ, rotation);

            Vector3 x_vel = Xworld * Vector3.Dot(velocity, Xworld);
            Vector3 y_vel = Yworld * Vector3.Dot(velocity, Yworld);
            Vector3 z_vel = Zworld * Vector3.Dot(velocity, Zworld);

            Vector3 xoy_vel = x_vel + y_vel;
            xoy_vel.Normalize();

            Vector3 xoz_vel = x_vel + z_vel;
            xoz_vel.Normalize();

            if (velocity.LengthSquared() > 1e-5)
            {
                // угол атаки
                alpha = Math.Asin(Clamp(Vector3.Dot(xoy_vel, -Yworld), -1.0, 1.0));
                // угол скольжения
                beta = Math.Asin(Clamp(Vector3.Dot(xoz_vel, Zworld), -1.0, 1.0));
            }
            else
            {
                alpha = 0.0;
                beta = 0.0;
            }
        }

        // Оси скоростной системы координат в нормальной
        public Vector3 Xaworld, Yaworld, Zaworld;
        Vector3 Xaworld_p, Yaworld_p, Zaworld_p;

        void calculate_acceleration(Vector3 velocity, Vector3 Yworld, Vector3 Zworld, double dyn_press, 
            double alpha, double beta, Vector3 control_surface, out Vector3 Xaworld, out Vector3 Yaworld, 
            out Vector3 Zaworld, out Vector3 acceleration)
        {
            Vector3 gravity = new Vector3(0.0, -g, 0.0);

            Xaworld = velocity;
            Xaworld.Normalize();

            Yaworld = Vector3.Cross(Zworld, velocity);
            Yaworld.Normalize();

            Zaworld = Vector3.Cross(velocity, Yworld);
            Zaworld.Normalize();

            Vector3 drag = -Xaworld * dyn_press * 
                (Cx0 + Cxa1 * alpha + Cxa2 * alpha * alpha + Cxb1 * beta + Cxb2 * beta * beta);
            Vector3 lift = Yaworld * dyn_press * (Cy0 + Cya * alpha + Cyc * control_surface.Z);
            Vector3 sideforce = -Zaworld * dyn_press * (Cz0 + Czb * beta + Cyc * control_surface.Y);

            acceleration = gravity + (drag + lift + sideforce) / mass;
        }

        void calculate_ang_acc(double dyn_press, Vector3 control_surface, Vector3 ang_vel, double alpha, double beta, 
            double speed, out Vector3 ang_acc)
        {
            ang_acc = new Vector3();
            ang_acc.X = dyn_press * (Mx0 + Mxcx * control_surface.X + Mxcy * control_surface.Y + Mxb * beta + Mxw * ang_vel.X / speed)
                / MOI.X;
            ang_acc.Y = dyn_press * (My0 + Myb * beta + Myc * control_surface.Y) / MOI.Y;
            ang_acc.Z = dyn_press * (Mz0 + Mza * alpha + Mzc * control_surface.Z) / MOI.Z;
        }

        // Численное интегрирование, явный метод Эйлера
        void euler_predict(double dt)
        {
            // центр масс
            position_predict = position + dt * velocity;
            // скорость
            velocity_predict = velocity + acceleration * dt;
            // угловая скорость
            ang_vel_predict = ang_vel + dt * ang_acc;

            // ориентация и угловая скорость
            //Vector3 ang_vel = Vector3.Transform(ang_vel, rotation);
            //Vector3 world_ang_acc = rotation.Value.Transform(ang_acc);
            Vector3 delta_rot = ang_vel * dt;

            if (delta_rot.LengthSquared() > 0.0)
                rotation_predict = Quaternion.CreateFromAxisAngle(delta_rot, -delta_rot.Length()) * rotation;
            else
                rotation_predict = rotation;
        }

        // Метод трапеций
        void trapezoidal_corrector(double dt)
        {
            // скорость
            velocity_predict = velocity + 0.5 * dt * (acceleration + acceleration_predict);
            // центр масс
            position_predict = position + 0.5 * dt * (velocity + velocity_predict);
            // угловая скорость
            ang_vel_predict = ang_vel + 0.5 * dt * (ang_acc + ang_acc_predict);

            // ориентация и угловая скорость
            //Vector3 world_ang_vel = Vector3.Transform(ang_vel, rotation);
            //Vector3 world_ang_vel_p = Vector3.Transform(ang_vel_predict, rotation_predict);
            Vector3 delta_rot = 0.5 * dt * (ang_vel + ang_vel_predict);

            if (delta_rot.LengthSquared() > 0.0)
                rotation_predict = Quaternion.CreateFromAxisAngle(delta_rot, -delta_rot.Length()) * rotation;
            else
                rotation_predict = rotation;
        }

        #endregion
    }
}
