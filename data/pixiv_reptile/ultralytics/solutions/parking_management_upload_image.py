def upload_image(self):
    """Upload an image and resize it to fit canvas."""
    self.image_path = filedialog.askopenfilename(filetypes=[('Image Files',
        '*.png;*.jpg;*.jpeg')])
    if not self.image_path:
        return
    self.image = Image.open(self.image_path)
    self.img_width, self.img_height = self.image.size
    aspect_ratio = self.img_width / self.img_height
    if aspect_ratio > 1:
        canvas_width = min(self.canvas_max_width, self.img_width)
        canvas_height = int(canvas_width / aspect_ratio)
    else:
        canvas_height = min(self.canvas_max_height, self.img_height)
        canvas_width = int(canvas_height * aspect_ratio)
    if self.canvas:
        self.canvas.destroy()
    self.canvas = self.tk.Canvas(self.master, bg='white', width=
        canvas_width, height=canvas_height)
    resized_image = self.image.resize((canvas_width, canvas_height), Image.
        LANCZOS)
    self.canvas_image = ImageTk.PhotoImage(resized_image)
    self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)
    self.canvas.pack(side=self.tk.BOTTOM)
    self.canvas.bind('<Button-1>', self.on_canvas_click)
    self.bounding_boxes = []
    self.current_box = []
