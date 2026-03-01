#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import messagebox, filedialog
import os
import subprocess
import threading
import time
import datetime
import queue
from pathlib import Path
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

# =================================================================
# [ KLine-Predictor: KRONOS INFERENCE TERMINAL ]
# GUI for running prediction inference using finetuned Kronos models
# =================================================================

class PredictorMatrixGUI(ttk.Window):
    def __init__(self):
        super().__init__(themename="cyborg")
        self.title("Kronos Predictor · 脑机推演终端")
        self.geometry("1100x860")
        self.minsize(1050, 800)
        
        try:
            self.createcommand('::tk::mac::ReopenApplication', self.deiconify)
        except Exception:
            pass
            
        # --- UI Colors & Styles (Flat Dark Gold) ---
        self.c_bg = "#080808"        
        self.c_panel = "#101010"     
        self.c_gold = "#F0B90B"      
        self.c_gold_dim = "#715A2B"  
        self.c_fg = "#E1C699"        
        self.c_green = "#00D47C"     
        self.c_red = "#FF3B30"       
        
        self.font_title = ("Menlo", 32, "bold")
        self.font_base = ("Menlo", 14)
        self.font_base_lg = ("Menlo", 16)
        self.font_log = ("Menlo", 13)

        # --- Paths & State ---
        self.root_dir = Path(__file__).resolve().parent  # 5-Kronos-Predictor-GUI
        self.suite_dir = self.root_dir.parent
        self.trainer_base_dir = self.suite_dir / "4-Kronos-Trainer-Base"
        self.models_dir = self.trainer_base_dir / "models"
        self.venv_python = self.trainer_base_dir / "venv" / "bin" / "python"
        
        self.target_csv_var = tk.StringVar(value="")
        
        # Hyperparameters (locked to x-matrix-kronos defaults)
        self.lookback_var = tk.IntVar(value=90)
        self.pred_len_var = tk.IntVar(value=10)
        self.temp_var = tk.DoubleVar(value=1.0)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.num_samples_var = tk.IntVar(value=1)
        
        self.process = None
        self.log_queue = queue.Queue()
        self.stop_requested = False
        
        self._setup_styles()
        self._build_ui()
        self.after(100, self._process_log_queue)

    def _setup_styles(self):
        style = ttk.Style()
        style.configure(".", font=self.font_base, background=self.c_bg, foreground=self.c_fg)
        
        style.configure("FlatGold.TButton", font=self.font_base_lg, background=self.c_bg, foreground=self.c_gold, bordercolor=self.c_gold, borderwidth=1)
        style.map("FlatGold.TButton", background=[("active", "#1A140B")], foreground=[("active", "#FFD700")])
        style.configure("FlatRed.TButton", font=self.font_base_lg, background=self.c_bg, foreground=self.c_red, bordercolor=self.c_red, borderwidth=1)
        style.map("FlatRed.TButton", background=[("active", "#1A0505")])
        style.configure("Hidden.Vertical.TScrollbar", background=self.c_gold_dim, troughcolor=self.c_bg, bordercolor=self.c_bg, relief="flat")
        style.map("Hidden.Vertical.TScrollbar", background=[("active", self.c_gold)])
        
        style.configure("TSpinbox", background=self.c_panel, foreground=self.c_gold)

    def _build_ui(self):
        self.configure(bg=self.c_bg)
        self.lift()
        self.attributes('-topmost', True)
        self.after(500, lambda: self.attributes('-topmost', False))
        os.system('''osascript -e 'tell application "System Events" to set frontmost of the first process whose unix id is %d to true' ''' % os.getpid())

        # --- HEADER ---
        header_frame = tk.Frame(self, bg=self.c_bg, pady=15)
        header_frame.pack(fill=X, padx=20)
        tk.Label(header_frame, text="KRONOS PREDICTOR · 脑机推演终端", font=self.font_title, fg=self.c_gold, bg=self.c_bg).pack(side=LEFT)
        self.status_sign = tk.Label(header_frame, text="系统就绪", font=("Menlo", 16, "bold"), fg=self.c_gold_dim, bg=self.c_bg)
        self.status_sign.pack(side=RIGHT, anchor=S)

        body_frame = tk.Frame(self, bg=self.c_bg)
        body_frame.pack(fill=BOTH, expand=True, padx=20, pady=(0, 20))
        
        # --- LEFT PANEL: CONTROLS ---
        left_panel = tk.Frame(body_frame, width=450, bg=self.c_bg)
        left_panel.pack(side=LEFT, fill=Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # 1. Mounts
        mount_lf = DashFrame(left_panel, title=" 推演数据与模型挂载 ", bg_color=self.c_bg, fg_color=self.c_gold, dash_color=self.c_gold_dim, font=("Menlo", 15, "bold"))
        mount_lf.pack(fill=X, pady=(0, 15))
        
        tk.Label(mount_lf.content, text="底层脑区装载目录:", font=self.font_base, fg=self.c_fg, bg=self.c_bg).pack(anchor=W)
        tk.Label(mount_lf.content, text=str(self.models_dir), font=("Menlo", 11), fg=self.c_gold_dim, bg=self.c_bg, wraplength=400, justify=LEFT).pack(anchor=W, pady=(0, 10))
        
        tk.Label(mount_lf.content, text="加载输入 K 线 (5分钟 CSV):", font=self.font_base, fg=self.c_fg, bg=self.c_bg).pack(anchor=W)
        src_fr = tk.Frame(mount_lf.content, bg=self.c_bg)
        src_fr.pack(fill=X, pady=(2, 5))
        tk.Entry(src_fr, textvariable=self.target_csv_var, font=self.font_log, bg=self.c_panel, fg=self.c_gold, relief="flat", highlightthickness=1, highlightbackground=self.c_gold_dim).pack(side=LEFT, fill=X, expand=True)
        ttk.Button(src_fr, text="打开", style="FlatGold.TButton", command=self.on_browse_csv).pack(side=RIGHT, padx=(5,0))
        
        # 2. Hyperparameters
        param_lf = DashFrame(left_panel, title=" 推演引擎配置 ", bg_color=self.c_bg, fg_color=self.c_gold, dash_color=self.c_gold_dim, font=("Menlo", 15, "bold"))
        param_lf.pack(fill=X, pady=(0, 15))
        
        self._add_param_row(param_lf.content, "历史回望视窗 (Lookback)", self.lookback_var, 10, 1000)
        self._add_param_row(param_lf.content, "未来预测步长 (Predict Length)", self.pred_len_var, 1, 500)
        self._add_param_row(param_lf.content, "推演温度 (Temperature)", self.temp_var, 0.1, 2.0, is_float=True)
        self._add_param_row(param_lf.content, "Top-P 采样边界", self.top_p_var, 0.1, 1.0, is_float=True)
        self._add_param_row(param_lf.content, "并行采样路径数 (Paths)", self.num_samples_var, 1, 10)

        # 3. Action Controls
        ctrl_lf = DashFrame(left_panel, title=" 神经网路操作面板 ", bg_color=self.c_bg, fg_color=self.c_gold, dash_color=self.c_gold_dim, font=("Menlo", 15, "bold"))
        ctrl_lf.pack(fill=X, pady=(0, 15))

        self.start_btn = ttk.Button(ctrl_lf.content, text="⚡ 启动脑机推演序列 (Infer)", style="FlatGold.TButton", command=self.on_start_click)
        self.start_btn.pack(fill=X, pady=(5, 10), ipady=5)
        
        self.stop_btn = ttk.Button(ctrl_lf.content, text="🛑 强制阻断推理进程", style="FlatRed.TButton", command=self.on_stop, state=DISABLED)
        self.stop_btn.pack(fill=X, pady=(0, 5), ipady=5)
        
        # 4. Result output
        out_lf = DashFrame(left_panel, title=" 预测产出目录 ", bg_color=self.c_bg, fg_color=self.c_green, dash_color=self.c_gold_dim, font=("Menlo", 15, "bold"))
        out_lf.pack(fill=BOTH, expand=True)
        
        self.out_dir = self.root_dir / "predictions"
        self.out_dir.mkdir(exist_ok=True)
        
        tk.Label(out_lf.content, text="导出路经:", font=self.font_base, fg=self.c_fg, bg=self.c_bg).pack(anchor=W)
        tk.Label(out_lf.content, text=str(self.out_dir), font=("Menlo", 11), fg=self.c_gold, bg=self.c_bg, wraplength=400, justify=LEFT).pack(anchor=W, pady=(2, 10))
        
        ttk.Button(out_lf.content, text="一键直达预测产出库", style="FlatGold.TButton", command=lambda: os.system(f'open "{self.out_dir}"')).pack(fill=X, pady=5)

        # --- RIGHT PANEL ---
        right_panel = tk.Frame(body_frame, bg=self.c_bg)
        right_panel.pack(side=LEFT, fill=BOTH, expand=True)
        
        term_lf = DashFrame(right_panel, title=" 隔离环境流式推演输出 (Inference Stream) ", bg_color=self.c_bg, fg_color=self.c_gold, dash_color=self.c_gold_dim, font=("Menlo", 15, "bold"))
        term_lf.pack(fill=BOTH, expand=True)
        
        txt_frame = tk.Frame(term_lf.content, bg=self.c_panel)
        txt_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)
        
        self.log_widget = tk.Text(txt_frame, font=self.font_log, bg=self.c_panel, fg="#A0A0A0", insertbackground=self.c_fg, wrap=WORD, borderwidth=0, highlightthickness=0, spacing1=4, spacing3=4)
        txt_scroll = ttk.Scrollbar(txt_frame, orient=tk.VERTICAL, command=self.log_widget.yview, style="Hidden.Vertical.TScrollbar")
        self.log_widget.configure(yscrollcommand=txt_scroll.set)
        
        txt_scroll.pack(side=RIGHT, fill=Y)
        self.log_widget.pack(side=LEFT, fill=BOTH, expand=True)
        
        # Colors for terminal
        self.log_widget.tag_config("sys", foreground=self.c_gold, font=("Menlo", 13, "bold"))
        self.log_widget.tag_config("err", foreground=self.c_red, font=("Menlo", 13, "bold"))
        self.log_widget.tag_config("succ", foreground=self.c_green, font=("Menlo", 13, "bold"))
        self.log_widget.tag_config("data", foreground="#00BFFF")
        self.log_widget.configure(state=DISABLED)

        # Footer
        self.status_bar = tk.Label(self, text=f"推理隔离环境: {self.venv_python}", bg=self.c_bg, fg=self.c_gold_dim, font=("Menlo", 11))
        self.status_bar.pack(side=BOTTOM, anchor=W, padx=20, pady=(0, 5))

    def _add_param_row(self, parent, label_text, var, from_, to_, is_float=False):
        row = tk.Frame(parent, bg=self.c_bg)
        row.pack(fill=X, pady=3)
        tk.Label(row, text=label_text, font=self.font_base, fg=self.c_fg, bg=self.c_bg).pack(side=LEFT)
        if is_float:
            sb = ttk.Spinbox(row, from_=from_, to=to_, increment=0.1, textvariable=var, width=6, font=self.font_base, style="TSpinbox")
        else:
            sb = ttk.Spinbox(row, from_=from_, to=to_, textvariable=var, width=6, font=self.font_base, style="TSpinbox")
        sb.pack(side=RIGHT)

    def gui_log(self, msg, level="info"):
        self.log_queue.put((msg, level))

    def _process_log_queue(self):
        try:
            while True:
                msg, level = self.log_queue.get_nowait()
                self.log_widget.configure(state=NORMAL)
                
                # Basic parsing for coloring
                if "Saving" in msg or "成功" in msg: level = "succ"
                elif "错误" in msg or "Error" in msg or "Traceback" in msg: level = "err"
                elif "Generating" in msg or "Step" in msg: level = "data"
                
                self.log_widget.insert(END, msg + "\n", level)
                self.log_widget.see(END)
                self.log_widget.configure(state=DISABLED)
        except queue.Empty:
            pass
        self.after(100, self._process_log_queue)

    def on_browse_csv(self):
        init_dir = str(self.suite_dir / "2-KLine-Resample" / "gui_out_5m")
        f = filedialog.askopenfilename(initialdir=init_dir, filetypes=[("CSV KLine Data", "*.csv")])
        if f:
            self.target_csv_var.set(f)

    def on_start_click(self):
        csv_path = self.target_csv_var.get().strip()
        if not csv_path or not os.path.exists(csv_path):
            messagebox.showerror("错误", "请先选择合法的待测 CSV 数据！")
            return
            
        if not self.venv_python.exists():
            messagebox.showerror("环境异常", f"找不到隔离环境的 Python 解释器:\n{self.venv_python}")
            return
            
        script_path = self.root_dir / "run_predictor_inference.py"
        
        self.stop_requested = False
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)
        self.status_sign.config(text="🧿 推演中...", fg=self.c_gold)
        
        # Clear log
        self.log_widget.configure(state=NORMAL)
        self.log_widget.delete('1.0', END)
        self.log_widget.configure(state=DISABLED)
        
        self.gui_log("[*] 启动脑区神经网路推演服务...", "sys")
        self.gui_log(f"[*] Target CSV: {csv_path}", "info")
        self.gui_log(f"[*] Lookback={self.lookback_var.get()}, Pred={self.pred_len_var.get()}, Temp={self.temp_var.get()}", "info")
        self.gui_log("--------------------------------------------------\n", "sys")
        
        threading.Thread(target=self._run_inference_process, args=(script_path, csv_path), daemon=True).start()

    def _run_inference_process(self, script_path, csv_path):
        env = os.environ.copy()
        env["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        args_list = [
            str(self.venv_python), "-u", str(script_path),
            "--csv", csv_path,
            "--out_dir", str(self.out_dir),
            "--lookback", str(self.lookback_var.get()),
            "--pred_len", str(self.pred_len_var.get()),
            "--temperature", str(self.temp_var.get()),
            "--top_p", str(self.top_p_var.get()),
            "--num_samples", str(self.num_samples_var.get())
        ]
        
        try:
            self.process = subprocess.Popen(
                args_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(self.root_dir),
                env=env
            )
            
            for line in iter(self.process.stdout.readline, ''):
                if self.stop_requested:
                    break
                self.gui_log(line.rstrip())
                
            self.process.stdout.close()
            return_code = self.process.wait()
            
            if self.stop_requested:
                self.gui_log("\n[!] 推演进程已被强制切断。", "warn")
                self.after(0, lambda: self.status_sign.config(text="阻断", fg=self.c_gold_dim))
            elif return_code == 0:
                self.gui_log("\n[+] 未来时光线谱已成功收容入库！", "succ")
                self.after(0, lambda: self.status_sign.config(text="推演完成", fg=self.c_green))
            else:
                self.gui_log(f"\n[-] 推演引擎异常熔断 (Code: {return_code})", "err")
                self.after(0, lambda: self.status_sign.config(text="异常终止", fg=self.c_red))
                
        except Exception as e:
            self.gui_log(f"[-] 进程启动失败: {e}", "err")
            self.after(0, lambda: self.status_sign.config(text="系统错误", fg=self.c_red))
            
        self.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        self.process = None

    def on_stop(self):
        if self.process:
            self.stop_requested = True
            self.gui_log("\n[!] 正在切断神经元连接 (SIGINT)...", "err")
            try:
                self.process.terminate()
            except Exception as e:
                self.gui_log(f"[-] 无法安全阻断: {e}", "err")

class DashFrame(tk.Frame):
    def __init__(self, master, title, bg_color, fg_color, dash_color, font, **kwargs):
        super().__init__(master, bg=bg_color, **kwargs)
        self.bg_color = bg_color
        self.dash_color = dash_color
        self.canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.content = tk.Frame(self, bg=bg_color)
        self.content.pack(fill=BOTH, expand=True, padx=12, pady=(25, 12)) 
        self.bind("<Configure>", self._draw)
        self.title_text = title
        self.fg_color = fg_color
        self.font = font
    def _draw(self, event=None):
        self.canvas.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10 or h < 10: return
        self.canvas.create_rectangle(2, 10, w-2, h-2, outline=self.dash_color, dash=(5, 5))
        self.canvas.create_rectangle(15, 0, 15 + len(self.title_text)*10, 20, fill=self.bg_color, outline="")
        self.canvas.create_text(20, 10, text=self.title_text, anchor="nw", font=self.font, fill=self.fg_color)

if __name__ == "__main__":
    try:
        app = PredictorMatrixGUI()
        app.mainloop()
    except Exception as e:
        print(f"\n[致命错误] GUI启动失败:")
        import traceback
        traceback.print_exc()
        input("\n按回车键退出 (Wait for input)...")
