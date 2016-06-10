using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Reflection;
using System.Windows.Forms.DataVisualization.Charting;

namespace ANNLanding
{
    public partial class MainForm : Form
    {
        public MainForm()
        {
            InitializeComponent();
            // инициализация списков данных для отображения
            string[] enum_list_names = typeof(Simulator.FlightData).GetEnumNames();
            comboBox1.Items.AddRange(enum_list_names);
            comboBox1.SelectedIndex = 0;
            comboBox2.Items.AddRange(enum_list_names);
            comboBox1.Items.Add("XY");
            comboBox2.Items.Add("XY");
            comboBox2.SelectedIndex = comboBox2.Items.Count - 1;
            // инициализация диалогов
            saveFileDialog1.InitialDirectory = Application.StartupPath;
            openFileDialog1.InitialDirectory = Application.StartupPath;
            // делегаты тренировки эмулятора
            chart_dlg_emul = new chart_delegate(report_emul);
            chart_dlg_regul = new chart_delegate(report_regul);
            // наборы кнопок
            start_buttons_emul.Add(button5);
            start_buttons_emul.Add(button6);
            start_buttons_emul.Add(button8);
            start_buttons_emul.Add(button9);
            start_buttons_emul.Add(button11);
            start_buttons_emul.Add(button13);

            start_buttons_regul.Add(button23);

            // datagridviews
            initDataGridViews();
        }

        Simulator simulator = new Simulator();

        #region Вкладка Модель

        // Кнопка "Симулировать" в первой вкладке
        private void button1_Click(object sender, EventArgs e)
        {
            double timeMax = 5.0;
            double.TryParse(textBox1.Text, out timeMax);

            double dt = 0.02;
            double.TryParse(textBox2.Text, out dt);

            double V = 100.0;
            double.TryParse(textBox3.Text, out V);

            double H = 500.0;
            double.TryParse(textBox4.Text, out H);

            double alpha_start = 5.0;
            double.TryParse(textBox5.Text, out alpha_start);

            double c_z = 0.0;
            double.TryParse(textBox6.Text, out c_z);

            double x = -100.0;
            double.TryParse(textBox7.Text, out x);

            IControlSystem csystem = null;
            Simulator.StateDelegate dlg = null;
            if (checkBox6.Checked)
            {
                csystem = new NeuralControlSystem(regulator, regulTrainer);
                dlg = (s, t, m) => { return s[0, 0] >= 0.0; };
            }
            else
                csystem = new EmulatorTrainer.DummyControlSystem(new XnaGeometry.Vector3(0.0, 0.0, c_z));

            simulator.SandboxInit(V, 0.0, H, x, alpha_start, c_z);
            if (checkBox4.Checked)
            {
                simulator.model.control_surface = new XnaGeometry.Vector3(0.0, 0.0, c_z);
                simulator.Simulate(csystem, dlg, timeMax, dt);
            }
            if (checkBox2.Checked)
            {
                simulator.SimulateAnn(emulator, csystem, dlg, timeMax, dt);
            }

            update_charts();

            // Отчёт об ошибке
            if (checkBox6.Checked)
            {
                double y_error = 0.0;
                double v_y_error = 0.0;
                double angv_error = 0.0;

                if (checkBox4.Checked)
                {
                    y_error = fin_Y - simulator.experiment_result[Simulator.FlightData.Altitude].Last();
                    v_y_error = fin_Vy - simulator.experiment_result[Simulator.FlightData.vel_y].Last();
                    angv_error = fin_angvel - simulator.experiment_result[Simulator.FlightData.ang_vel_z].Last();

                    label47.Text = y_error.ToString("G5");
                    label45.Text = v_y_error.ToString("G5");
                    label43.Text = angv_error.ToString("G5");
                }

                if (checkBox2.Checked)
                {
                    y_error = fin_Y - simulator.emulator_result[Simulator.FlightData.Altitude].Last();
                    v_y_error = fin_Vy - simulator.emulator_result[Simulator.FlightData.vel_y].Last();
                    angv_error = fin_angvel - simulator.emulator_result[Simulator.FlightData.ang_vel_z].Last();

                    label33.Text = y_error.ToString("G5");
                    label34.Text = v_y_error.ToString("G5");
                    label36.Text = angv_error.ToString("G5");
                }
            }
            else
            {
                label47.Text = "0.0";
                label45.Text = "0.0";
                label43.Text = "0.0";
                label33.Text = "0.0";
                label34.Text = "0.0";
                label36.Text = "0.0";
            }
        }

        void update_charts()
        {
            combobox_switch(comboBox1, chart1);
            combobox_switch(comboBox2, chart2);
        }

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            combobox_switch(comboBox1, chart1);
        }

        void combobox_switch(ComboBox box, Chart chart)
        {
            if (box.SelectedIndex < 0)
            {
                chart.Series[0].Points.Clear();
                chart.Series[1].Points.Clear();
                return;
            }
            // check for trajectory
            if (box.SelectedIndex == box.Items.Count - 1)
            {
                chart.Series[0].Name = "траектория";
                chart.Series[0].Points.Clear();
                if (checkBox4.Checked)
                {
                    if (simulator.experiment_result.ContainsKey(Simulator.FlightData.Altitude))
                    {
                        var y = simulator.experiment_result[Simulator.FlightData.Altitude];
                        var x = simulator.experiment_result[Simulator.FlightData.X];
                        for (int i = 0; i < x.Count; i++)
                            chart.Series[0].Points.AddXY(x[i], y[i]);
                    }
                }
                chart.Series[1].Points.Clear();
                if (checkBox2.Checked)
                {
                    if (simulator.emulator_result.ContainsKey(Simulator.FlightData.Altitude))
                    {
                        var y = simulator.emulator_result[Simulator.FlightData.Altitude];
                        var x = simulator.emulator_result[Simulator.FlightData.X];
                        for (int i = 0; i < x.Count; i++)
                            chart.Series[1].Points.AddXY(x[i], y[i]);
                    }
                }
            }
            else
            {
                string enum_name = (string)box.Items[box.SelectedIndex];
                Simulator.FlightData enum_type = (Simulator.FlightData)Enum.Parse(typeof(Simulator.FlightData), enum_name);
                chart.Series[0].Name = enum_name;
                chart.Series[0].Points.Clear();
                if (checkBox4.Checked)
                {
                    if (simulator.experiment_result.ContainsKey(enum_type))
                    {
                        var res = simulator.experiment_result[enum_type];
                        var time = simulator.experiment_result[Simulator.FlightData.time];
                        for (int i = 0; i < time.Count; i++)
                            chart.Series[0].Points.AddXY(time[i], res[i]);
                    }
                }
                chart.Series[1].Points.Clear();
                if (checkBox2.Checked)
                {
                    if (simulator.emulator_result.ContainsKey(enum_type))
                    {
                        var res = simulator.emulator_result[enum_type];
                        var time = simulator.emulator_result[Simulator.FlightData.time];
                        for (int i = 0; i < time.Count; i++)
                            chart.Series[1].Points.AddXY(time[i], res[i]);
                    }
                }
            }
        }

        private void comboBox2_SelectedIndexChanged(object sender, EventArgs e)
        {
            combobox_switch(comboBox2, chart2);
        }

        #endregion

        #region Эмулятор

        TansigAnn emulator = new TansigAnn(6, 2, new int[2] { 20, 6 }, true, 0.5, 0.1);

        private void button3_Click(object sender, EventArgs e)
        {
            int neuron_count = 20;
            int.TryParse(textBox9.Text, out neuron_count);

            int layer_count = 1;
            int.TryParse(textBox14.Text, out layer_count);

            int[] layers = new int[layer_count + 1];
            for (int i = 0; i < layer_count; i++)
                layers[i] = neuron_count;
            layers[layer_count] = 6;        // 6 выходов

            double rng_weight_diap = 0.5;
            double.TryParse(textBox10.Text, out rng_weight_diap);

            double rng_bias_diap = 0.5;
            double.TryParse(textBox20.Text, out rng_bias_diap);

            emulator = new TansigAnn(6, layer_count + 1, layers, checkBox1.Checked, rng_weight_diap, rng_bias_diap);

            double dt = 0.05;
            double.TryParse(textBox11.Text, out dt);
            //emul_trainer.init_emulator_weights(emulator, dt, simulator.model);
        }

        private void button2_Click(object sender, EventArgs e)
        {
            var result = saveFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
                emulator.Serialize(saveFileDialog1.FileName);
        }

        private void button4_Click(object sender, EventArgs e)
        {
            var result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
                emulator = TansigAnn.Deserialize(openFileDialog1.FileName);
        }

        EmulatorTrainer emul_trainer = new EmulatorTrainer();
        Task training_task;

        // Создать набор
        private void button5_Click(object sender, EventArgs e)
        {
            int set_size = 1000;
            int.TryParse(textBox13.Text, out set_size);
            double dt = 0.05;
            double.TryParse(textBox11.Text, out dt);

            emul_trainer.init_rng_borders();
            emul_trainer.generate_training_points(set_size, dt);

            if (set_size > 0)
            {
                button6.Enabled = true;
                button8.Enabled = true;
                button9.Enabled = true;
                button11.Enabled = true;
            }
            else
            {
                button6.Enabled = false;
                button8.Enabled = false;
                button9.Enabled = false;
                button11.Enabled = false;
            }
        }

        delegate void chart_delegate(int epoch, double t_error, double g_error, double v_error);
        chart_delegate chart_dlg_emul, chart_dlg_regul;

        void report_emul(int epoch, double t_error, double g_error, double v_error)
        {
            chart3.Series[0].Points.AddXY(epoch, t_error);
            chart3.Series[1].Points.AddXY(epoch, g_error);
            chart3.Series[2].Points.AddXY(epoch, v_error);
        }

        void report_regul(int epoch, double t_error, double g_error, double v_error)
        {
            chart5.Series[0].Points.AddXY(epoch, t_error);
            //chart5.Series[1].Points.AddXY(epoch, g_error);
            //chart5.Series[2].Points.AddXY(epoch, v_error);
        }

        void clear_chart3()
        {
            foreach (var series in chart3.Series)
                series.Points.Clear();
        }

        void report_delegate_emul(int epoch, double t_error, double g_error, double v_error)
        {
            chart3.BeginInvoke(chart_dlg_emul, new object[] { epoch, t_error, g_error, v_error });
        }

        bool stop_flag_emul = false;

        private void button7_Click(object sender, EventArgs e)
        {
            stop_flag_emul = true;
            training_task.Wait();
            stop_flag_emul = false;
        }

        // Сигналы остановки тренировки
        delegate void void_dlg();

        // Бесконечная тренировка
        private void button8_Click(object sender, EventArgs e)
        {
            int set_size = emul_trainer.PairsCount;
            double dt = 0.05;
            double.TryParse(textBox11.Text, out dt);

            training_emul_start();
            foreach (var series in chart3.Series)
                series.Points.Clear();

            emul_trainer.generate_validation_points(1000, dt);
            emul_trainer.generate_training_points(set_size, dt, 20.0, 0.0, false);

            training_task = new Task(() =>
            {
                infinite_training_func(set_size, dt);
                signal_training_emul_end();
            });
            training_task.Start();
        }

        void infinite_training_func(int set_size_start, double dt)
        {
            int set_size = set_size_start;
            int fail_descend = 0;
            while (!stop_flag_emul)
            {
                emul_trainer.train_emulator_points(emulator, report_delegate_emul, ref stop_flag_emul);
                double new_valid_error = chart3.Series["validation_error"].Points.Last().YValues[0];
                double old_valid_error = chart3.Series["validation_error"].Points[0].YValues[0];
                if (new_valid_error >= old_valid_error)
                {
                    // Не удалось спуститься, вернёмся к сохранённой сети
                    //approximator = approx_res.Copy();
                    fail_descend++;
                }
                else
                {
                    // спуск удачен
                    fail_descend = 1;
                    emulator.Serialize("ann_autosave.txt");
                }
                // очистим график
                if (!stop_flag_emul)
                    chart3.BeginInvoke(new void_dlg(clear_chart3));
                // если слишком много неспусков, расширим набор тренировки
                if (/*fail_descend > 0 && */!stop_flag_emul)
                {
                    fail_descend = 0;
                    set_size *= 2;
                    set_size = Math.Min(set_size, 1000000);
                }
                if (!stop_flag_emul)
                    emul_trainer.generate_training_points(set_size, dt, 20.0, 0.0, false);
            }
        }

        List<Button> start_buttons_emul = new List<Button>();

        void signal_training_emul_end()
        {
            foreach (var btn in start_buttons_emul)
                btn.BeginInvoke(new void_dlg(() => { btn.Enabled = true;}));
            button7.BeginInvoke(new void_dlg(() => { button7.Enabled = false; }));
        }

        void training_emul_start()
        {
            foreach (var btn in start_buttons_emul)
                btn.Enabled = false;
            button7.Enabled = true;
        }

        // Тренировка точки
        private void button6_Click(object sender, EventArgs e)
        {
            training_emul_start();
            foreach (var series in chart3.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                emul_trainer.train_emulator_points(emulator, report_delegate_emul, ref stop_flag_emul);
                signal_training_emul_end();
            });
            training_task.Start();
        }

        // Patricle swarm
        private void button9_Click(object sender, EventArgs e)
        {
            double rng_weight_diap = 0.5;
            double.TryParse(textBox10.Text, out rng_weight_diap);

            double rng_bias_diap = 0.5;
            double.TryParse(textBox20.Text, out rng_bias_diap);

            double inertia = 0.5;
            double.TryParse(textBox17.Text, out inertia);

            double c1 = 0.5;
            double.TryParse(textBox18.Text, out c1);

            double c2 = 0.5;
            double.TryParse(textBox19.Text, out c2);

            double max_v = 0.5;
            double.TryParse(textBox16.Text, out max_v);

            int p_count = 50;
            int.TryParse(textBox15.Text, out p_count);

            training_emul_start();
            foreach (var series in chart3.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                emul_trainer.train_pso_points(emulator, report_delegate_emul, ref stop_flag_emul,
                    p_count, max_v, inertia, c1, c2, rng_weight_diap, rng_bias_diap, checkBox3.Checked);
                signal_training_emul_end();
            });
            training_task.Start();
        }

        // стохастическая тренировка
        private void button11_Click(object sender, EventArgs e)
        {
            double v = 0.5;
            double.TryParse(textBox23.Text, out v);

            double acc = 0.5;
            double.TryParse(textBox24.Text, out acc);

            double decc = 0.5;
            double.TryParse(textBox25.Text, out decc);

            training_emul_start();
            foreach (var series in chart3.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                emul_trainer.train_emulator_stoh(emulator, report_delegate_emul, ref stop_flag_emul, v, acc, decc);
                signal_training_emul_end();
            });
            training_task.Start();
        }

        List<TansigAnn.TrainingPair> debug_training;
        List<TansigAnn.TrainingPair> debug_gen;
        List<TansigAnn.TrainingPair> debug_valid;
        
        TansigAnn debug_ann = new TansigAnn(1, 2, new int[2] { 3, 1 }, true, 1.0, 1.0);

        // Создать синус
        private void button12_Click(object sender, EventArgs e)
        {
            List<TansigAnn.TrainingPair> debug_pairs = new List<TansigAnn.TrainingPair>();
            debug_ann = new TansigAnn(1, 2, new int[2] { 5, 1 }, true, 1.0, 1.0);
            double dx = 0.01;
            double k = 5.0;
            double.TryParse(textBox26.Text, out k);
            chart4.Series[0].Points.Clear();
            chart4.Series[1].Points.Clear();
            for (double x = -1.0; x <= 1.0; x += dx)
            {
                Matrix input = new Matrix(1, 1);
                Matrix output = new Matrix(1, 1);
                input[0, 0] = x;
                output[0, 0] = Math.Sin(k * x);
                TansigAnn.TrainingPair pair = new TansigAnn.TrainingPair(input, output);
                debug_pairs.Add(pair);
                chart4.Series[0].Points.AddXY(x, output[0, 0]);
            }
            // вывести на чарт

            int count = debug_pairs.Count;
            int v_count = (int)(count * 0.1);
            int g_count = (int)(count * 0.2);

            Random rng = new Random();
            debug_gen = new List<TansigAnn.TrainingPair>();
            debug_valid = new List<TansigAnn.TrainingPair>();
            while (g_count > 0)
            {
                int index = rng.Next() % debug_pairs.Count;
                TansigAnn.TrainingPair pair = debug_pairs[index];
                debug_pairs.RemoveAt(index);
                debug_gen.Add(pair);
                g_count--;
            }
            while (v_count > 0)
            {
                int index = rng.Next() % debug_pairs.Count;
                TansigAnn.TrainingPair pair = debug_pairs[index];
                debug_pairs.RemoveAt(index);
                debug_valid.Add(pair);
                v_count--;
            }
            debug_training = debug_pairs;

            button13.Enabled = true;
        }

        private void button13_Click(object sender, EventArgs e)
        {
            int neuron_count = 20;
            int.TryParse(textBox9.Text, out neuron_count);

            debug_ann = new TansigAnn(1, 2, new int[2] { neuron_count, 1 }, true, 1.0, 1.0);
            training_emul_start();
            foreach (var series in chart3.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                debug_ann.train_CGBP(debug_training, debug_gen, debug_valid, Matrix.IdentityMatrix(1, 1), 1e-10, 10000,
                    report_delegate_emul, ref stop_flag_emul, 200, 100);
                signal_training_emul_end();
            });
            training_task.Start();
        }

        private void button14_Click(object sender, EventArgs e)
        {
            double dx = 0.01;
            chart4.Series[1].Points.Clear();
            for (double x = -1.0; x <= 1.0; x += dx)
            {
                Matrix input = new Matrix(1, 1);
                input[0, 0] = x;
                Matrix output = debug_ann.eval(input);
                chart4.Series[1].Points.AddXY(x, output[0, 0]);
            }
        }

        private void button15_Click(object sender, EventArgs e)
        {
            emul_trainer.update_output_normalization();
        }

        private void button16_Click(object sender, EventArgs e)
        {
            var result = saveFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
                Rescale.serialize(saveFileDialog1.FileName);
        }

        private void button17_Click(object sender, EventArgs e)
        {
            var result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
            {
                Rescale.deserialize(openFileDialog1.FileName);
                emul_trainer.init_rng_borders();
            }
        }

        #endregion Эмулятор


        TansigAnn regulator = new TansigAnn(7, 2, new int[2] { 3, 1 }, true, 1.0, 1.0);
        RegulatorTrainer regulTrainer = new RegulatorTrainer();

        // Сброс
        private void button30_Click(object sender, EventArgs e)
        {
            int neuron_count = 4;
            int.TryParse(textBox42.Text, out neuron_count);

            int layer_count = 1;
            int.TryParse(textBox38.Text, out layer_count);

            int[] layers = new int[layer_count + 1];
            for (int i = 0; i < layer_count; i++)
                layers[i] = neuron_count;
            layers[layer_count] = 1;        // 1 выход

            double rng_weight_diap = 0.5;
            double.TryParse(textBox41.Text, out rng_weight_diap);

            double rng_bias_diap = 0.5;
            double.TryParse(textBox32.Text, out rng_bias_diap);

            regulator = new TansigAnn(7, layer_count + 1, layers, checkBox5.Checked, rng_weight_diap, rng_bias_diap);
        }

        // сохранить
        private void button31_Click(object sender, EventArgs e)
        {
            var result = saveFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
                regulator.Serialize(saveFileDialog1.FileName);
        }

        // загрузить
        private void button29_Click(object sender, EventArgs e)
        {
            var result = openFileDialog1.ShowDialog();
            if (result == DialogResult.OK)
                regulator = TansigAnn.Deserialize(openFileDialog1.FileName);
        }

        // создать набор
        private void button28_Click(object sender, EventArgs e)
        {
            int set_size = 100;
            int.TryParse(textBox31.Text, out set_size);
            //double dt = 0.05;
            //double.TryParse(textBox40.Text, out dt);
            double Xstart = -200.0;
            double.TryParse(textBox12.Text, out Xstart);
            double Ystart = 25.0;
            double.TryParse(textBox30.Text, out Ystart);
            double Ydevia = 5.0;
            double.TryParse(textBox39.Text, out Ydevia);
            double cz_start = 0.5;
            double.TryParse(textBox21.Text, out cz_start);

            regulTrainer.generate_training_processes(set_size, Xstart, Ystart, Ydevia, 5.0 * PlaneModel.dgr2rad, cz_start, 32.0, 0.0);
            //emul_trainer.generate_training_processes(proc_count, maxT, dt);

            if (set_size > 0)
            {
                button23.Enabled = true;
            }
            else
            {
                button23.Enabled = false;
            }
        }

        bool stop_flag_regul = false;

        void report_delegate_regul(int epoch, double t_error, double g_error, double v_error)
        {
            chart5.BeginInvoke(chart_dlg_regul, new object[] { epoch, t_error, g_error, v_error });
        }

        // Тренировать процессы
        private void button23_Click(object sender, EventArgs e)
        {
            double speed = 1e-2;
            double.TryParse(textBox22.Text, out speed);

            updateRegulWeightFromGrid();
            training_regul_start();
            foreach (var series in chart5.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                regulTrainer.train_regulator_processes(regulator, emulator, report_delegate_regul, ref stop_flag_regul, speed);
                signal_training_regul_end();
            });
            training_task.Start();
        }

        List<Button> start_buttons_regul = new List<Button>();

        void training_regul_start()
        {
            foreach (var btn in start_buttons_regul)
                btn.Enabled = false;
            button26.Enabled = true;
        }

        void signal_training_regul_end()
        {
            foreach (var btn in start_buttons_regul)
                btn.BeginInvoke(new void_dlg(() => { btn.Enabled = true; }));
            button26.BeginInvoke(new void_dlg(() => { button26.Enabled = false; }));
        }

        private void button26_Click(object sender, EventArgs e)
        {
            stop_flag_regul = true;
            training_task.Wait();
            stop_flag_regul = false;
        }

        // Претренировка
        private void button10_Click(object sender, EventArgs e)
        {
            double des_val = 0.5;
            double.TryParse(textBox21.Text, out des_val);

            training_regul_start();
            foreach (var series in chart5.Series)
                series.Points.Clear();
            training_task = new Task(() =>
            {
                regulTrainer.pretrain_regulator_dumb(regulator, report_delegate_regul, 4000, des_val, 100, ref stop_flag_regul);
                signal_training_regul_end();
            });
            training_task.Start();
        }

        void initDataGridViews()
        {
            dataGridView1.Rows.Add(4);

            dataGridView1.Rows[0].Cells[0].Value = "координата X";
            dataGridView1.Rows[0].Cells[1].Value = "0,0";
            dataGridView1.Rows[0].Cells[2].Value = "0,0";

            dataGridView1.Rows[1].Cells[0].Value = "координата Y";
            dataGridView1.Rows[1].Cells[1].Value = "0,0";
            dataGridView1.Rows[1].Cells[2].Value = "1,0";

            dataGridView1.Rows[2].Cells[0].Value = "Vy";
            dataGridView1.Rows[2].Cells[1].Value = "-0,5";
            dataGridView1.Rows[2].Cells[2].Value = "1,0";

            dataGridView1.Rows[3].Cells[0].Value = "угл скорость";
            dataGridView1.Rows[3].Cells[1].Value = "0,0";
            dataGridView1.Rows[3].Cells[2].Value = "1,0";

            dataGridView2.Rows.Add(3);

            dataGridView2.Rows[0].Cells[0].Value = "Vy";
            dataGridView2.Rows[0].Cells[1].Value = "0,0";
            dataGridView2.Rows[0].Cells[2].Value = "0,001";

            dataGridView2.Rows[1].Cells[0].Value = "угл скорость";
            dataGridView2.Rows[1].Cells[1].Value = "0,0";
            dataGridView2.Rows[1].Cells[2].Value = "0,00001";

            dataGridView2.Rows[2].Cells[0].Value = "угол атаки";
            dataGridView2.Rows[2].Cells[1].Value = "0,0";
            dataGridView2.Rows[2].Cells[2].Value = "0,00001";
        }

        double fin_X = 0.0;
        double fin_Y = 0.0;
        double fin_Vy = -0.5;
        double fin_angvel = 0.0;

        void updateRegulWeightFromGrid()
        {
            // Конечное состояние            
            double.TryParse(dataGridView1.Rows[0].Cells[1].Value as string, out fin_X);            
            double.TryParse(dataGridView1.Rows[1].Cells[1].Value as string, out fin_Y);
            double.TryParse(dataGridView1.Rows[2].Cells[1].Value as string, out fin_Vy);            
            double.TryParse(dataGridView1.Rows[3].Cells[1].Value as string, out fin_angvel);

            double fin_X_weight = 0.0;
            double.TryParse(dataGridView1.Rows[0].Cells[2].Value as string, out fin_X_weight);
            double fin_Y_weight = 0.0;
            double.TryParse(dataGridView1.Rows[1].Cells[2].Value as string, out fin_Y_weight);
            double fin_Vy_weight = -0.5;
            double.TryParse(dataGridView1.Rows[2].Cells[2].Value as string, out fin_Vy_weight);
            double fin_angvel_weight = 0.0;
            double.TryParse(dataGridView1.Rows[3].Cells[2].Value as string, out fin_angvel_weight);

            // введём в тренер
            regulTrainer.desired_end_state[0, 0] = Rescale.norm(fin_X, RegulatorTrainer.Xmin, RegulatorTrainer.Xmax);
            regulTrainer.desired_end_state[1, 0] = Rescale.norm(fin_Y, RegulatorTrainer.Ymin, RegulatorTrainer.Ymax);
            regulTrainer.desired_end_state[3, 0] = Rescale.norm_input(fin_Vy, 1);
            regulTrainer.desired_end_state[4, 0] = Rescale.norm_input(fin_angvel, 2);

            regulTrainer.end_state_weights[0, 0] = fin_X_weight;
            regulTrainer.end_state_weights[1, 0] = fin_Y_weight;
            regulTrainer.end_state_weights[3, 0] = fin_Vy_weight;
            regulTrainer.end_state_weights[4, 0] = fin_angvel_weight;

            // По траектории

            double trj_Vy = 0.0;
            double.TryParse(dataGridView2.Rows[0].Cells[1].Value as string, out trj_Vy);
            double trj_angvel = 0.0;
            double.TryParse(dataGridView2.Rows[1].Cells[1].Value as string, out trj_angvel);
            double trj_alpha = 0.0;
            double.TryParse(dataGridView2.Rows[2].Cells[1].Value as string, out trj_alpha);

            double trj_Vy_weight = 1e-4;
            double.TryParse(dataGridView2.Rows[0].Cells[2].Value as string, out trj_Vy_weight);
            double trj_angvel_weight = 1e-4;
            double.TryParse(dataGridView2.Rows[1].Cells[2].Value as string, out trj_angvel_weight);
            double trj_alpha_weight = 1e-4;
            double.TryParse(dataGridView2.Rows[2].Cells[2].Value as string, out trj_alpha_weight);

            // введём в тренер
            regulTrainer.desired_transit_state[3, 0] = Rescale.norm_input(trj_Vy, 1);
            regulTrainer.desired_transit_state[4, 0] = Rescale.norm_input(trj_angvel, 2);
            regulTrainer.desired_transit_state[5, 0] = Rescale.norm_input(trj_alpha, 3);

            regulTrainer.transit_state_weights[3, 0] = trj_Vy_weight;
            regulTrainer.transit_state_weights[4, 0] = trj_angvel_weight;
            regulTrainer.transit_state_weights[5, 0] = trj_alpha_weight;
        }
    }
}
