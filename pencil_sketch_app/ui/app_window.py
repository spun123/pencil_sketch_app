from __future__ import annotations

import os
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

from pencil_sketch_app.config.settings import (
    APP_NAME,
    AI_REFERENCE_MODES,
    DEFAULT_EDIT_PROMPT,
    DEFAULT_OPENAI_MODEL,
    LOCAL_AI_LOW_MEMORY_DEFAULT,
    MODE_AUTO_AI,
    MODE_INSTANTID,
    MODE_IPADAPTER,
    MODE_LOCAL,
    MODE_OPENAI,
    OPENAI_API_KEY_ENV,
    OUTPUT_DIR,
    PREVIEW_MAX_H,
    PREVIEW_MAX_W,
    WINDOW_MIN_H,
    WINDOW_MIN_W,
)
from pencil_sketch_app.core.image_io import cv_read_image_unicode, cv_write_image_unicode, safe_filename
from pencil_sketch_app.pipelines.router import process_image_by_mode


class PencilSketchApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1360x900")
        self.root.minsize(WINDOW_MIN_W, WINDOW_MIN_H)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.input_path: Path | None = None
        self.original_bgr: np.ndarray | None = None
        self.result_bgr: np.ndarray | None = None
        self.saved_result_path: Path | None = None

        self.preview_original_tk = None
        self.preview_result_tk = None

        self._build_style()
        self._build_ui()
        self._set_status(f"Готово. Папка результатов: {OUTPUT_DIR}")

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass

        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 9))
        style.configure("Info.TLabel", font=("Segoe UI", 10))
        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=12)
        main_frame.pack(fill="both", expand=True)

        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(top_frame, text=APP_NAME, style="Title.TLabel").pack(side="left")

        controls_frame = ttk.LabelFrame(main_frame, text="Управление", padding=12)
        controls_frame.pack(fill="x", pady=(0, 10))

        buttons_frame = ttk.Frame(controls_frame)
        buttons_frame.pack(fill="x", pady=(0, 10))

        ttk.Button(buttons_frame, text="Добавить фото", style="Action.TButton", command=self.load_image).pack(side="left", padx=(0, 8))
        ttk.Button(buttons_frame, text="Преобразовать", style="Action.TButton", command=self.process_image).pack(side="left", padx=(0, 8))
        ttk.Button(buttons_frame, text="Сохранить как", style="Action.TButton", command=self.save_as).pack(side="left", padx=(0, 8))
        ttk.Button(buttons_frame, text="Открыть папку результатов", command=self.open_results_folder).pack(side="left", padx=(0, 8))
        ttk.Button(buttons_frame, text="Очистить", command=self.clear_all).pack(side="left")

        mode_frame = ttk.Frame(controls_frame)
        mode_frame.pack(fill="x", pady=(0, 10))

        self.processing_mode_var = tk.StringVar(value=MODE_LOCAL)
        ttk.Label(mode_frame, text="Режим обработки:").pack(side="left", padx=(0, 8))
        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.processing_mode_var,
            state="readonly",
            width=28,
            values=[MODE_LOCAL, MODE_OPENAI, MODE_AUTO_AI, MODE_INSTANTID, MODE_IPADAPTER],
        )
        self.mode_combo.pack(side="left", padx=(0, 12))
        self.mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

        self.ai_similarity_var = tk.IntVar(value=75)
        self.low_memory_mode_var = tk.BooleanVar(value=LOCAL_AI_LOW_MEMORY_DEFAULT)
        ttk.Label(mode_frame, text="Похожесть к фото:").pack(side="left", padx=(0, 8))
        self.ai_similarity_scale = ttk.Scale(
            mode_frame,
            from_=10,
            to=100,
            orient="horizontal",
            command=lambda value: self.ai_similarity_var.set(int(float(value))),
        )
        self.ai_similarity_scale.set(self.ai_similarity_var.get())
        self.ai_similarity_scale.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(mode_frame, textvariable=self.ai_similarity_var, width=4).pack(side="left", padx=(0, 12))
        self.low_memory_check = ttk.Checkbutton(
            mode_frame,
            text="Экономия памяти",
            variable=self.low_memory_mode_var,
        )
        self.low_memory_check.pack(side="left")

        openai_frame = ttk.Frame(controls_frame)
        openai_frame.pack(fill="x", pady=(0, 10))

        self.openai_key_var = tk.StringVar(value=OPENAI_API_KEY_ENV)
        ttk.Label(openai_frame, text="API-ключ OpenAI:").pack(side="left", padx=(0, 8))
        self.openai_key_entry = ttk.Entry(openai_frame, textvariable=self.openai_key_var, width=46, show="*")
        self.openai_key_entry.pack(side="left", padx=(0, 8))

        self.show_key_var = tk.BooleanVar(value=False)
        self.show_key_check = ttk.Checkbutton(
            openai_frame,
            text="Показать ключ",
            variable=self.show_key_var,
            command=self.toggle_key_visibility,
        )
        self.show_key_check.pack(side="left")

        prompt_frame = ttk.LabelFrame(controls_frame, text="Настройка генерации", padding=10)
        prompt_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(prompt_frame, text="Текстовое задание:").pack(anchor="w")
        self.prompt_text = tk.Text(prompt_frame, height=5, wrap="word", font=("Segoe UI", 10))
        self.prompt_text.pack(fill="x", pady=(4, 8))
        self.prompt_text.insert("1.0", DEFAULT_EDIT_PROMPT)

        options_online = ttk.Frame(prompt_frame)
        options_online.pack(fill="x")
        self.openai_options_frame = options_online

        self.openai_model_var = tk.StringVar(value=DEFAULT_OPENAI_MODEL)
        self.quality_var = tk.StringVar(value="high")
        self.size_var = tk.StringVar(value="1024x1024")

        ttk.Label(options_online, text="Модель:").pack(side="left", padx=(0, 6))
        ttk.Combobox(options_online, textvariable=self.openai_model_var, state="readonly", width=18, values=["gpt-image-1", "gpt-image-1-mini"]).pack(side="left", padx=(0, 12))
        ttk.Label(options_online, text="Качество:").pack(side="left", padx=(0, 6))
        ttk.Combobox(options_online, textvariable=self.quality_var, state="readonly", width=10, values=["high", "medium", "low", "auto"]).pack(side="left", padx=(0, 12))
        ttk.Label(options_online, text="Размер:").pack(side="left", padx=(0, 6))
        ttk.Combobox(options_online, textvariable=self.size_var, state="readonly", width=12, values=["1024x1024", "1536x1024", "1024x1536", "auto"]).pack(side="left", padx=(0, 12))

        local_frame = ttk.LabelFrame(main_frame, text="Локальные параметры", padding=12)
        local_frame.pack(fill="x", pady=(0, 10))
        self.local_frame = local_frame

        self.line_brightness_var = tk.IntVar(value=168)
        self.contour_low_var = tk.IntVar(value=60)
        self.contour_high_var = tk.IntVar(value=140)
        self.noise_cleaning_var = tk.IntVar(value=28)
        self.line_thickness_var = tk.IntVar(value=1)
        self.keep_extra_details_var = tk.BooleanVar(value=False)

        local_options = ttk.Frame(local_frame)
        local_options.pack(fill="x")

        col1 = ttk.Frame(local_options)
        col1.pack(side="left", fill="x", expand=True, padx=(0, 12))
        col2 = ttk.Frame(local_options)
        col2.pack(side="left", fill="x", expand=True, padx=(0, 12))
        col3 = ttk.Frame(local_options)
        col3.pack(side="left", fill="x", expand=True)

        self._make_labeled_scale(col1, "Светлота линий (графитовый цвет)", self.line_brightness_var, 135, 210)
        self._make_labeled_scale(col1, "Толщина линий", self.line_thickness_var, 1, 3)
        self._make_labeled_scale(col2, "Чувствительность к слабым контурам", self.contour_low_var, 20, 160)
        self._make_labeled_scale(col2, "Чувствительность к сильным контурам", self.contour_high_var, 60, 240)
        self._make_labeled_scale(col3, "Степень очистки мелких деталей", self.noise_cleaning_var, 5, 80)
        ttk.Checkbutton(col3, text="Добавить немного второстепенных деталей", variable=self.keep_extra_details_var).pack(anchor="w", pady=(12, 0))

        preview_wrap = ttk.Frame(main_frame)
        preview_wrap.pack(fill="both", expand=True)

        left_panel = ttk.LabelFrame(preview_wrap, text="Исходное изображение", padding=10)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right_panel = ttk.LabelFrame(preview_wrap, text="Результат", padding=10)
        right_panel.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.original_info = ttk.Label(left_panel, text="Файл не загружен", style="Info.TLabel")
        self.original_info.pack(anchor="w", pady=(0, 8))
        self.result_info = ttk.Label(right_panel, text="Результат ещё не создан", style="Info.TLabel")
        self.result_info.pack(anchor="w", pady=(0, 8))

        self.original_canvas = tk.Label(left_panel, bg="#ffffff", relief="solid", bd=1)
        self.original_canvas.pack(fill="both", expand=True)
        self.result_canvas = tk.Label(right_panel, bg="#ffffff", relief="solid", bd=1)
        self.result_canvas.pack(fill="both", expand=True)

        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill="x", pady=(10, 0))
        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(bottom_frame, textvariable=self.status_var).pack(side="left")

        self.on_mode_changed()

    def _make_labeled_scale(self, parent: ttk.Frame, text: str, variable: tk.IntVar, frm: int, to: int) -> None:
        wrap = ttk.Frame(parent)
        wrap.pack(fill="x", pady=3)
        ttk.Label(wrap, text=text).pack(anchor="w")
        line = ttk.Frame(wrap)
        line.pack(fill="x")
        scale = ttk.Scale(line, from_=frm, to=to, orient="horizontal", command=lambda value, v=variable: v.set(int(float(value))))
        scale.set(variable.get())
        scale.pack(side="left", fill="x", expand=True)
        ttk.Label(line, textvariable=variable, width=5).pack(side="left", padx=(8, 0))

    def _set_children_state(self, widget: tk.Widget, state: str) -> None:
        for child in widget.winfo_children():
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
            self._set_children_state(child, state)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def toggle_key_visibility(self) -> None:
        self.openai_key_entry.configure(show="" if self.show_key_var.get() else "*")

    def on_mode_changed(self, event=None) -> None:
        mode = self.processing_mode_var.get()
        online_mode = mode == MODE_OPENAI
        local_art_mode = mode == MODE_LOCAL
        local_ai_mode = mode in AI_REFERENCE_MODES

        openai_state = "normal" if online_mode else "disabled"
        self.openai_key_entry.configure(state=openai_state)
        self.show_key_check.configure(state=openai_state)

        self.prompt_text.configure(state="normal")
        if not online_mode and not local_ai_mode:
            self.prompt_text.configure(state="disabled")

        for child in self.openai_options_frame.winfo_children():
            try:
                child.configure(state=openai_state)
            except tk.TclError:
                pass

        self.local_frame.configure(text="Локальные параметры" if local_art_mode else "Локальные параметры скетча")
        self._set_children_state(self.local_frame, "normal" if local_art_mode else "disabled")

        try:
            self.ai_similarity_scale.configure(state="normal" if local_ai_mode else "disabled")
            self.low_memory_check.configure(state="normal" if local_ai_mode else "disabled")
        except tk.TclError:
            pass

    def load_image(self) -> None:
        filetypes = [("Изображения", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff;*.webp"), ("Все файлы", "*.*")]
        file_path = filedialog.askopenfilename(title="Выберите фото или рисунок", filetypes=filetypes)
        if not file_path:
            return

        img = cv_read_image_unicode(file_path)
        if img is None:
            messagebox.showerror("Ошибка", "Не удалось открыть выбранный файл.")
            return

        self.input_path = Path(file_path)
        self.original_bgr = img
        self.result_bgr = None
        self.saved_result_path = None
        self.original_info.config(text=f"Файл: {self.input_path.name}")
        self.result_info.config(text="Результат ещё не создан")
        self._show_preview(self.original_canvas, self.original_bgr, preview_type="original")
        self.result_canvas.configure(image="")
        self.result_canvas.image = None
        self._set_status(f"Загружено: {self.input_path}")

    def process_image(self) -> None:
        if self.original_bgr is None:
            messagebox.showwarning("Нет файла", "Сначала добавьте фото или рисунок.")
            return

        try:
            prompt = self.prompt_text.get("1.0", "end").strip() or DEFAULT_EDIT_PROMPT
            line_art_settings = {
                "contour_low": self.contour_low_var.get(),
                "contour_high": self.contour_high_var.get(),
                "line_brightness": self.line_brightness_var.get(),
                "noise_cleaning": self.noise_cleaning_var.get(),
                "line_thickness": self.line_thickness_var.get(),
                "keep_extra_details": self.keep_extra_details_var.get(),
            }
            mode = self.processing_mode_var.get()
            self._set_status(f"Преобразование: {mode}...")

            result, effective_mode = process_image_by_mode(
                image_bgr=self.original_bgr,
                mode=mode,
                prompt=prompt,
                openai_api_key=self.openai_key_var.get(),
                openai_model=self.openai_model_var.get(),
                openai_quality=self.quality_var.get(),
                openai_size=self.size_var.get(),
                similarity_strength=self.ai_similarity_var.get(),
                low_memory_mode=self.low_memory_mode_var.get(),
                line_art_settings=line_art_settings,
            )

            self.result_bgr = result
            self._show_preview(self.result_canvas, self.result_bgr, preview_type="result")
            auto_saved = self._auto_save_result()
            self.saved_result_path = auto_saved
            self.result_info.config(text=f"Автосохранение: {auto_saved.name}")
            self._set_status(f"Готово. Режим: {effective_mode}. Сохранено: {auto_saved}")
        except Exception as exc:
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Не удалось преобразовать изображение.\n\n{exc}")
            self._set_status("Ошибка преобразования")

    def _auto_save_result(self) -> Path:
        if self.result_bgr is None:
            raise RuntimeError("Нет результата для сохранения.")
        src_name = safe_filename(self.input_path.stem if self.input_path else "image")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"{src_name}_sketch_{timestamp}.png"
        ok = cv_write_image_unicode(out_path, self.result_bgr)
        if not ok:
            raise IOError(f"Не удалось сохранить файл: {out_path}")
        return out_path

    def save_as(self) -> None:
        if self.result_bgr is None:
            messagebox.showwarning("Нет результата", "Сначала выполните преобразование изображения.")
            return

        default_name = safe_filename(self.input_path.stem) + "_sketch.png" if self.input_path else "result.png"
        out_path = filedialog.asksaveasfilename(
            title="Сохранить как",
            defaultextension=".png",
            initialfile=default_name,
            initialdir=str(OUTPUT_DIR),
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("BMP", "*.bmp")],
        )
        if not out_path:
            return

        ok = cv_write_image_unicode(out_path, self.result_bgr)
        if not ok:
            messagebox.showerror("Ошибка", "Не удалось сохранить файл.")
            return

        backup_path = None
        try:
            ext = Path(out_path).suffix or ".png"
            backup_name = f"manual_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            backup_path = OUTPUT_DIR / backup_name
            shutil.copy2(out_path, backup_path)
        except Exception:
            backup_path = None

        self.saved_result_path = Path(out_path)
        message = f"Файл сохранён:\n{out_path}"
        if backup_path is not None:
            message += f"\n\nКопия также сохранена в:\n{backup_path}"
        messagebox.showinfo("Сохранение завершено", message)
        self.result_info.config(text=f"Сохранено как: {Path(out_path).name}")
        self._set_status(f"Сохранено: {out_path}")

    def open_results_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(OUTPUT_DIR))
        elif sys.platform == "darwin":
            os.system(f'open "{OUTPUT_DIR}"')
        else:
            os.system(f'xdg-open "{OUTPUT_DIR}"')

    def clear_all(self) -> None:
        self.input_path = None
        self.original_bgr = None
        self.result_bgr = None
        self.saved_result_path = None
        self.original_canvas.configure(image="")
        self.original_canvas.image = None
        self.result_canvas.configure(image="")
        self.result_canvas.image = None
        self.original_info.config(text="Файл не загружен")
        self.result_info.config(text="Результат ещё не создан")
        self._set_status("Очищено")

    def _show_preview(self, widget: tk.Label, image_bgr: np.ndarray, preview_type: str) -> None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        scale = min(PREVIEW_MAX_W / w, PREVIEW_MAX_H / h, 1.0)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil_img = Image.fromarray(resized)
        tk_img = ImageTk.PhotoImage(pil_img)
        widget.configure(image=tk_img)
        widget.image = tk_img
        if preview_type == "original":
            self.preview_original_tk = tk_img
        else:
            self.preview_result_tk = tk_img

    def on_close(self) -> None:
        self.root.destroy()
