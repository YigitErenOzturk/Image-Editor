import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

import transforms

class ImageEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Simple Image Editör")
        self.root.geometry("1100x650")

        self.original_bgr: np.ndarray | None = None
        self.processed_bgr: np.ndarray | None = None

        self._build_ui()

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        btn_load = tk.Button(top, text="Load Image (FR1)", command=self.load_image, width=18)
        btn_load.pack(side=tk.LEFT, padx=5)

        btn_save = tk.Button(top, text="Export Result (FR4)", command=self.save_image, width=18)
        btn_save.pack(side=tk.LEFT, padx=5)

        btn_reset = tk.Button(top, text="Reset", command=self.reset, width=10)
        btn_reset.pack(side=tk.LEFT, padx=5)

        # Filter buttons 
        filters = tk.LabelFrame(self.root, text="Transformations (FR2)")
        filters.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        tk.Button(filters, text="Grayscale", command=self.apply_grayscale, width=14).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(filters, text="Blur", command=self.apply_blur, width=14).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(filters, text="Edge (Canny)", command=self.apply_edge, width=14).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(filters, text="Rotate 90°", command=self.apply_rotate, width=14).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(filters, text="Flip Horizontal", command=self.apply_flip, width=14).pack(side=tk.LEFT, padx=5, pady=5)

        # Sliders for brightness and contrast 
        sliders = tk.LabelFrame(self.root, text="Brightness / Contrast (FR2)")
        sliders.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        self.brightness_var = tk.IntVar(value=0)
        self.contrast_var = tk.IntVar(value=0)

        tk.Label(sliders, text="Brightness").pack(side=tk.LEFT, padx=(10, 5))
        self.brightness_scale = tk.Scale(
            sliders, from_=-100, to=100, orient=tk.HORIZONTAL, length=300,
            variable=self.brightness_var, command=lambda _v: self.apply_brightness_contrast_live()
        )
        self.brightness_scale.pack(side=tk.LEFT, padx=5)

        tk.Label(sliders, text="Contrast").pack(side=tk.LEFT, padx=(20, 5))
        self.contrast_scale = tk.Scale(
            sliders, from_=-100, to=100, orient=tk.HORIZONTAL, length=300,
            variable=self.contrast_var, command=lambda _v: self.apply_brightness_contrast_live()
        )
        self.contrast_scale.pack(side=tk.LEFT, padx=5)

        # Before and After Section
        main = tk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = tk.LabelFrame(main, text="Before")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right = tk.LabelFrame(main, text="After ")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.before_label = tk.Label(left, text="Load an image to start.")
        self.before_label.pack(fill=tk.BOTH, expand=True)

        self.after_label = tk.Label(right, text="Apply transformations to see result.")
        self.after_label.pack(fill=tk.BOTH, expand=True)

        self.status = tk.Label(self.root, text="Ready.", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------- Helpers ----------
    def _set_status(self, msg: str):
        self.status.config(text=msg)

    def _bgr_to_tk(self, img_bgr: np.ndarray, max_w: int = 520, max_h: int = 520) -> ImageTk.PhotoImage:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)

        w, h = pil.size
        scale = min(max_w / w, max_h / h, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        pil = pil.resize((new_w, new_h))

        return ImageTk.PhotoImage(pil)

    def _refresh_views(self):
        if self.original_bgr is not None:
            tk_img = self._bgr_to_tk(self.original_bgr)
            self.before_label.configure(image=tk_img, text="")
            self.before_label.image = tk_img

        if self.processed_bgr is not None:
            tk_img2 = self._bgr_to_tk(self.processed_bgr)
            self.after_label.configure(image=tk_img2, text="")
            self.after_label.image = tk_img2

    def _ensure_loaded(self) -> bool:
        if self.original_bgr is None:
            messagebox.showwarning("No image", "Please load an image first.")
            return False
        return True

    def _set_processed(self, img_bgr: np.ndarray, msg: str):
        self.processed_bgr = img_bgr
        self._refresh_views()
        self._set_status(msg)

    # ---------- FR1 ----------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the selected file. Direction should only have English Letters.")
            return

        self.original_bgr = img
        self.processed_bgr = img.copy()

        # reset sliders
        self.brightness_var.set(0)
        self.contrast_var.set(0)

        self._refresh_views()
        self._set_status(f"Loaded: {path}")

    # ---------- FR4 ----------
    def save_image(self):
        if self.processed_bgr is None:
            messagebox.showwarning("Nothing to save", "No processed image to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPG", "*.jpg *.jpeg"), ("BMP", "*.bmp"), ("WEBP", "*.webp")]
        )
        if not path:
            return

        ok = cv2.imwrite(path, self.processed_bgr)
        if not ok:
            messagebox.showerror("Error", "Failed to save the image.")
            return
        self._set_status(f"Exported: {path}")

    def reset(self):
        if not self._ensure_loaded():
            return
        self.processed_bgr = self.original_bgr.copy()
        self.brightness_var.set(0)
        self.contrast_var.set(0)
        self._refresh_views()
        self._set_status("Reset to original.")

    # ---------- FR2: Transformations ----------
    def apply_grayscale(self):
        if not self._ensure_loaded():
            return
        out = transforms.to_grayscale(self.processed_bgr.copy())
        self._set_processed(out, "Applied: Grayscale")

    def apply_blur(self):
        if not self._ensure_loaded():
            return
        out = transforms.gaussian_blur(self.processed_bgr.copy(), ksize=9)
        self._set_processed(out, "Applied: Gaussian Blur")

    def apply_edge(self):
        if not self._ensure_loaded():
            return
        out = transforms.edge_canny(self.processed_bgr.copy(), t1=60, t2=160)
        self._set_processed(out, "Applied: Canny Edge Detection")

    def apply_rotate(self):
        if not self._ensure_loaded():
            return
        out = transforms.rotate_90(self.processed_bgr.copy())
        self._set_processed(out, "Applied: Rotate 90°")

    def apply_flip(self):
        if not self._ensure_loaded():
            return
        out = transforms.flip_horizontal(self.processed_bgr.copy())
        self._set_processed(out, "Applied: Flip Horizontal")

    def apply_brightness_contrast_live(self):
        if not self._ensure_loaded():
            return
        # Apply relative to original to avoid compounding noise
        base = self.original_bgr.copy()
        b = self.brightness_var.get()
        c = self.contrast_var.get()
        out = transforms.adjust_brightness_contrast(base, brightness=b, contrast=c)
        self._set_processed(out, f"Applied: Brightness={b}, Contrast={c}")

def main():
    root = tk.Tk()
    app = ImageEditorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
