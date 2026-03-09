import os
import math
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


APP_TITLE = "Skill Peak Clip Cutter"
DEFAULT_EXPORT_DIR = "clips_out"


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round((seconds - int(seconds)) * 1000))
    total = int(seconds)
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    if ms == 1000:
        total += 1
        ms = 0
        s = total % 60
        m = (total // 60) % 60
        h = total // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class VideoCutterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1100x760")
        self.root.minsize(980, 700)

        self.video_path = None
        self.export_dir = os.path.abspath(DEFAULT_EXPORT_DIR)
        os.makedirs(self.export_dir, exist_ok=True)

        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.duration = 0.0
        self.current_frame = 0
        self.current_time = 0.0

        self.in_time = None
        self.out_time = None

        self.playing = False
        self.preview_after_id = None
        self.current_photo = None
        self.export_index = 1

        self._build_ui()
        self._bind_shortcuts()
        self._set_status("Åbn en video for at starte.")

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Button(top, text="Åbn video", command=self.open_video).pack(side="left")
        ttk.Button(top, text="Vælg export-mappe", command=self.choose_export_dir).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Set IN (I)", command=self.set_in_point).pack(side="left", padx=(20, 0))
        ttk.Button(top, text="Set OUT (O)", command=self.set_out_point).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Nulstil points", command=self.clear_points).pack(side="left", padx=(8, 0))
        ttk.Button(top, text="Eksportér klip", command=self.export_clip_threaded).pack(side="left", padx=(20, 0))

        self.mode_var = tk.StringVar(value="copy")
        ttk.Label(top, text="Mode:").pack(side="left", padx=(20, 6))
        ttk.Radiobutton(top, text="Lossless / fast", variable=self.mode_var, value="copy").pack(side="left")
        ttk.Radiobutton(top, text="Præcis / re-encode", variable=self.mode_var, value="precise").pack(side="left", padx=(8, 0))

        preview_wrap = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        preview_wrap.pack(fill="both", expand=True)

        self.preview_label = tk.Label(
            preview_wrap,
            text="Ingen video valgt",
            bg="black",
            fg="white",
            anchor="center",
            relief="sunken",
        )
        self.preview_label.pack(fill="both", expand=True)

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        self.play_button = ttk.Button(controls, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")

        ttk.Button(controls, text="<< 1 frame", command=lambda: self.step_frames(-1)).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="1 frame >>", command=lambda: self.step_frames(1)).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="-1 sek", command=lambda: self.step_seconds(-1)).pack(side="left", padx=(16, 0))
        ttk.Button(controls, text="+1 sek", command=lambda: self.step_seconds(1)).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="-5 sek", command=lambda: self.step_seconds(-5)).pack(side="left", padx=(16, 0))
        ttk.Button(controls, text="+5 sek", command=lambda: self.step_seconds(5)).pack(side="left", padx=(8, 0))

        self.time_var = tk.StringVar(value="00:00:00.000 / 00:00:00.000")
        ttk.Label(controls, textvariable=self.time_var).pack(side="right")

        slider_wrap = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        slider_wrap.pack(fill="x")

        self.timeline_var = tk.DoubleVar(value=0.0)
        self.timeline = ttk.Scale(
            slider_wrap,
            from_=0.0,
            to=1.0,
            variable=self.timeline_var,
            orient="horizontal",
            command=self.on_timeline_drag,
        )
        self.timeline.pack(fill="x")

        info = ttk.Frame(self.root, padding=10)
        info.pack(fill="x")

        self.file_var = tk.StringVar(value="Fil: -")
        self.in_var = tk.StringVar(value="IN: -")
        self.out_var = tk.StringVar(value="OUT: -")
        self.export_var = tk.StringVar(value=f"Export-mappe: {self.export_dir}")

        ttk.Label(info, textvariable=self.file_var).pack(anchor="w")
        ttk.Label(info, textvariable=self.in_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(info, textvariable=self.out_var).pack(anchor="w", pady=(2, 0))
        ttk.Label(info, textvariable=self.export_var).pack(anchor="w", pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", padding=8).pack(fill="x", side="bottom")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<space>", lambda e: self.toggle_play())
        self.root.bind("<Left>", lambda e: self.step_frames(-1))
        self.root.bind("<Right>", lambda e: self.step_frames(1))
        self.root.bind("<Shift-Left>", lambda e: self.step_seconds(-1))
        self.root.bind("<Shift-Right>", lambda e: self.step_seconds(1))
        self.root.bind("i", lambda e: self.set_in_point())
        self.root.bind("o", lambda e: self.set_out_point())
        self.root.bind("e", lambda e: self.export_clip_threaded())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def on_close(self) -> None:
        self.playing = False
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def choose_export_dir(self) -> None:
        folder = filedialog.askdirectory(initialdir=self.export_dir)
        if folder:
            self.export_dir = folder
            os.makedirs(self.export_dir, exist_ok=True)
            self.export_var.set(f"Export-mappe: {self.export_dir}")

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Vælg video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.load_video(path)

    def load_video(self, path: str) -> None:
        if self.cap is not None:
            self.cap.release()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Fejl", "Kunne ikke åbne videoen.")
            return

        self.video_path = path
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if self.fps <= 1e-6:
            self.fps = 30.0
        self.duration = (self.total_frames / self.fps) if self.total_frames > 0 else 0.0
        self.current_frame = 0
        self.current_time = 0.0
        self.in_time = None
        self.out_time = None
        self.playing = False
        self.play_button.config(text="Play")

        self.timeline.configure(to=max(self.duration, 1.0))
        self.timeline_var.set(0.0)
        self.file_var.set(f"Fil: {os.path.basename(path)}")
        self._update_markers()
        self._read_and_show_frame(0)
        self._set_status("Video indlæst.")

    def _update_markers(self) -> None:
        self.in_var.set(f"IN: {format_seconds(self.in_time) if self.in_time is not None else '-'}")
        self.out_var.set(f"OUT: {format_seconds(self.out_time) if self.out_time is not None else '-'}")
        self.time_var.set(f"{format_seconds(self.current_time)} / {format_seconds(self.duration)}")

    def _read_and_show_frame(self, frame_index: int) -> None:
        if self.cap is None:
            return
        frame_index = max(0, min(frame_index, max(self.total_frames - 1, 0)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if not ok:
            return

        self.current_frame = frame_index
        self.current_time = self.current_frame / self.fps if self.fps else 0.0
        self.timeline_var.set(self.current_time)
        self._update_markers()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_w = max(self.preview_label.winfo_width(), 640)
        label_h = max(self.preview_label.winfo_height(), 360)

        h, w = frame_rgb.shape[:2]
        scale = min(label_w / w, label_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        image = Image.fromarray(resized)
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.current_photo, text="")

    def on_timeline_drag(self, _value: str) -> None:
        if self.cap is None:
            return
        if self.playing:
            self.playing = False
            self.play_button.config(text="Play")
        target_time = self.timeline_var.get()
        frame_index = int(round(target_time * self.fps))
        self._read_and_show_frame(frame_index)

    def toggle_play(self) -> None:
        if self.cap is None:
            return
        self.playing = not self.playing
        self.play_button.config(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_loop()

    def _play_loop(self) -> None:
        if not self.playing or self.cap is None:
            return

        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.playing = False
            self.play_button.config(text="Play")
            return

        self._read_and_show_frame(next_frame)
        delay = max(1, int(1000 / max(self.fps, 1.0)))
        self.preview_after_id = self.root.after(delay, self._play_loop)

    def step_frames(self, amount: int) -> None:
        if self.cap is None:
            return
        self.playing = False
        self.play_button.config(text="Play")
        self._read_and_show_frame(self.current_frame + amount)

    def step_seconds(self, seconds: float) -> None:
        if self.cap is None:
            return
        self.playing = False
        self.play_button.config(text="Play")
        target_time = self.current_time + seconds
        frame_index = int(round(target_time * self.fps))
        self._read_and_show_frame(frame_index)

    def set_in_point(self) -> None:
        if self.cap is None:
            return
        self.in_time = self.current_time
        if self.out_time is not None and self.out_time < self.in_time:
            self.out_time = self.in_time
        self._update_markers()
        self._set_status(f"IN sat til {format_seconds(self.in_time)}")

    def set_out_point(self) -> None:
        if self.cap is None:
            return
        self.out_time = self.current_time
        if self.in_time is not None and self.out_time < self.in_time:
            self.in_time = self.out_time
        self._update_markers()
        self._set_status(f"OUT sat til {format_seconds(self.out_time)}")

    def clear_points(self) -> None:
        self.in_time = None
        self.out_time = None
        self._update_markers()
        self._set_status("IN/OUT nulstillet.")

    def export_clip_threaded(self) -> None:
        if self.video_path is None:
            messagebox.showwarning("Ingen video", "Åbn en video først.")
            return
        if self.in_time is None or self.out_time is None:
            messagebox.showwarning("Mangler points", "Sæt både IN og OUT først.")
            return
        if self.out_time <= self.in_time:
            messagebox.showwarning("Ugyldigt klip", "OUT skal være større end IN.")
            return
        if shutil.which("ffmpeg") is None:
            messagebox.showerror(
                "FFmpeg mangler",
                "FFmpeg blev ikke fundet i PATH.\n\nInstaller FFmpeg og prøv igen.",
            )
            return

        threading.Thread(target=self.export_clip, daemon=True).start()

    def export_clip(self) -> None:
        start = self.in_time
        end = self.out_time
        basename = f"clip_{self.export_index:03d}_{format_seconds(start).replace(':', '-').replace('.', '_')}_to_{format_seconds(end).replace(':', '-').replace('.', '_')}.mp4"
        out_path = os.path.join(self.export_dir, basename)

        mode = self.mode_var.get()

        if mode == "copy":
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", format_seconds(start),
                "-to", format_seconds(end),
                "-i", self.video_path,
                "-c", "copy",
                out_path,
            ]
        else:
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", format_seconds(start),
                "-to", format_seconds(end),
                "-i", self.video_path,
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "192k",
                out_path,
            ]

        self.root.after(0, lambda: self._set_status("Eksporterer klip..."))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                err = result.stderr[-1000:] if result.stderr else "Ukendt FFmpeg-fejl"
                self.root.after(0, lambda: messagebox.showerror("Eksport fejlede", err))
                self.root.after(0, lambda: self._set_status("Eksport fejlede."))
                return

            self.export_index += 1
            self.root.after(0, lambda: self._set_status(f"Gemt: {out_path}"))
            self.root.after(0, lambda: messagebox.showinfo("Færdig", f"Klip gemt:\n{out_path}"))
        except Exception as exc:
            self.root.after(0, lambda: messagebox.showerror("Fejl", str(exc)))
            self.root.after(0, lambda: self._set_status("Eksport fejlede."))


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    app = VideoCutterApp(root)
    root.mainloop()
