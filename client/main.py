import tkinter as tk
import os

def install_libraries():
    os.system('pip install tk tkintermapview pillow requests python-dotenv')

def show_installation_tutorial():
    root = tk.Tk()
    root.title("Library Installation Tutorial")
    
    label = tk.Label(root, text="Greetings! Our program requires certain libraries to be installed in order for it to work, there was an error installing so please do it manually!")
    label.pack(padx=10, pady=10)
    
    command_label = tk.Label(root, text="To install the library: Open Command Prompt --? Type pip intall tk tkintermapview requests python-dotenv Once that is done, just restart this program and it should work!")
    command_label.pack(padx=10, pady=10)
    
    root.mainloop()

try:
    import tkinter as tk
    from tkinter import font, PhotoImage, Label, messagebox, filedialog
    import tkinter.simpledialog as simpledialog
    from tkintermapview import TkinterMapView
    from PIL import Image as PILImage
    from PIL import ImageTk, Image
    import os,base64, requests, threading, time
    from io import BytesIO
    from dotenv import load_dotenv
    import re
except ImportError:
    install_libraries()
    try:
        import re
        import tkinter as tk
        from tkinter import font, PhotoImage, Label, messagebox, filedialog
        import tkinter.simpledialog as simpledialog
        from tkintermapview import TkinterMapView
        from PIL import Image as PILImage
        from PIL import ImageTk
        import os,base64, requests, threading, time
        from io import BytesIO
        from dotenv import load_dotenv
    except ImportError:
        show_installation_tutorial()

class ReLeafApp:
    def __init__(self, root):
        load_dotenv()
        self.root, self.active_index = root, 0
        self.root.title("ReLeaf - Advanced GUI")
        self.root.geometry("1112x600")
        self.root.resizable(False, False)
        self.font_rasa = ('Sans Serif Collection', 12)
        self.font_dubai = ('Courier New Greek', 12)
        self.control_panel_color, self.button_active_color = "#758A48", "#899D5E"
        self.button_hover_color, self.content_bg_color = "#B07C57", "#FFFFFF"
        self.border_color = "#916A4F"
        self.file_path = ""
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        try:
            self.font_raleway = ("Raleway", 12)
            self.font_raleway_bold = ("Raleway", 16)
        except tk.TclError:
            self.font_raleway = font.nametofont("TkDefaultFont")
            self.font_raleway_bold = self.font_raleway.copy()
            self.font_raleway_bold.configure(size=16, weight="bold")

        self.root.configure(bg=self.control_panel_color)
        self.content_area = tk.Frame(self.root, bg=self.content_bg_color, width=900, height=600)
        self.content_area.place(x=212, y=0)

        tk.Label(self.root, text="ReLeaf", font=self.font_raleway_bold, bg=self.control_panel_color, fg="white").place(x=0, y=30, width=212)

        self.buttons =  ["Map", "Contact", "AI", "About"]
        self.icons = [PhotoImage(file=f"images/{name.lower()}.png") for name in self.buttons]
        self.buttons_frame = []
        self.setup_navigation()

        self.drawing = self.is_searching = False
        self.start_coords = self.end_coords = self.current_rectangle = None
        self.markers = []
        self.debounce_delay, self.last_search_time = 0.3, time.time()
        self.search_thread = None
        
        self.change_page(self.active_index)

    def setup_navigation(self):
        for index, name in enumerate(self.buttons):
            frame = tk.Frame(self.root, bg=self.control_panel_color)
            frame.place(x=0, y=100 + (index * 70), width=212, height=70)

            rectangle = tk.Frame(frame, bg="white", width=7, height=70)
            rectangle.place(x=0, y=0)
            rectangle.place_forget()

            icon_label = tk.Label(frame, image=self.icons[index], bg=self.control_panel_color)
            icon_label.place(x=25, rely=0.5, anchor="center")
            
            label = tk.Label(frame, text=name, font=self.font_raleway, bg=self.control_panel_color, fg="white")
            label.place(x=70, rely=0.5, anchor="w")

            for widget in [frame, label, icon_label]:
                widget.bind("<Enter>", lambda e, idx=index: self.on_hover(idx))
                widget.bind("<Leave>", lambda e, idx=index: self.on_leave(idx))
                widget.bind("<Button-1>", lambda e, idx=index: self.change_page(idx))

            rectangle.bind("<Button-1>", lambda e, idx=index: self.change_page(idx))
            self.buttons_frame.append((frame, label, rectangle, icon_label))

    def on_hover(self, index):
        if index != self.active_index:
            frame, label, _, icon_label = self.buttons_frame[index]
            for widget in [frame, label, icon_label]: widget.config(bg=self.button_hover_color)

    def on_leave(self, index):
        if index != self.active_index:
            frame, label, _, icon_label = self.buttons_frame[index]
            for widget in [frame, label, icon_label]: widget.config(bg=self.control_panel_color)

    def change_page(self, index):
        prev_frame, prev_label, prev_rectangle, prev_icon = self.buttons_frame[self.active_index]
        for widget in [prev_frame, prev_label, prev_icon]: widget.config(bg=self.control_panel_color)
        prev_rectangle.place_forget()

        self.active_index = index
        new_frame, new_label, new_rectangle, new_icon = self.buttons_frame[index]
        for widget in [new_frame, new_label, new_icon]: widget.config(bg=self.button_active_color)
        new_rectangle.place(x=0, y=0)

        for widget in self.content_area.winfo_children(): widget.destroy()
        {
            0: self.load_map_page, 
            1: self.show_contact_page, 
            2: self.show_ai_page, 
            3: self.show_about_page
        }[index]()

    def load_image(self, image_name): return ImageTk.PhotoImage(Image.open(os.path.join("images", image_name)))

    def create_rounded_rectangle(self, canvas):
        return lambda x1, y1, x2, y2, radius, **kwargs: canvas.create_polygon(x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius, x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2, x1, y2, x1, y2-radius, x1, y1+radius, x1, y1, smooth=True, **kwargs)

    def create_button(self, parent, width, height, text, image_name, command, x=None, y=None, relx=None, rely=None):
        frame = tk.Frame(parent, bg=self.content_bg_color)
        if x is not None:
            frame.place(x=x, y=y, width=width, height=height)
        else:
            frame.place(relx=relx, y=y, width=width, height=height, anchor="n")

        canvas = tk.Canvas(frame, width=width, height=height, bg=self.content_bg_color, highlightthickness=0)
        canvas.pack()
        
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, width, height, height/2, fill=self.control_panel_color, width=0)

        icon_image = self.load_image(image_name)
        icon = tk.Label(canvas, image=icon_image, bg=self.control_panel_color)
        icon.image = icon_image

        text_label = tk.Label(canvas, text=text, font=self.font_raleway, fg="white", bg=self.control_panel_color)

        total_width = icon_image.width() + text_label.winfo_reqwidth() + 5
        start_x = (width - total_width) / 2
        icon.place(x=start_x, rely=0.5, anchor="w")
        text_label.place(x=start_x + icon_image.width() + 5, rely=0.5, anchor="w")

        for widget in [canvas, icon, text_label]:
            widget.bind("<Enter>", lambda e: self.on_button_hover(canvas, icon, text_label))
            widget.bind("<Leave>", lambda e: self.on_button_leave(canvas, icon, text_label))
            widget.bind("<Button-1>", lambda e: command())

        return frame

    def on_button_hover(self, canvas, icon, text):
        canvas.delete("all")
        canvas.create_rounded_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), canvas.winfo_height()/2, fill=self.button_hover_color, width=0)
        icon.configure(bg=self.button_hover_color)
        text.configure(bg=self.button_hover_color)

    def on_button_leave(self, canvas, icon, text):
        canvas.delete("all")
        canvas.create_rounded_rectangle(0, 0, canvas.winfo_width(), canvas.winfo_height(), canvas.winfo_height()/2, fill=self.control_panel_color, width=0)
        icon.configure(bg=self.control_panel_color)
        text.configure(bg=self.control_panel_color)

    def load_map_page(self):
        tk.Frame(self.content_area, bg=self.border_color, width=702, height=2).place(relx=0.5, y=20, anchor="n")
        tk.Label(self.content_area, text="View Map", font=self.font_raleway_bold, bg=self.content_bg_color, fg=self.control_panel_color).place(relx=0.5, y=30, anchor="n")
        self.map_view = TkinterMapView(self.content_area, width=602, height=300)
        self.map_view.place(relx=0.5, y=80, anchor="n")
        self.map_view.set_tile_server("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", max_zoom=22)
        self.map_view.add_left_click_map_command(self.on_map_click)

        search_container = tk.Frame(self.content_area, bg=self.content_bg_color)
        search_container.place(relx=0.5, y=380, anchor="n")

        search_frame = tk.Frame(search_container, bg=self.content_bg_color, width=300, height=55)
        search_frame.pack(side="top")
        search_frame.pack_propagate(False)

        canvas = tk.Canvas(search_frame, width=280, height=50, bg=self.content_bg_color, highlightthickness=0)
        canvas.place(relx=0.5, rely=0.5, anchor="center")
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, 280, 50, 25, fill="#EFEFEF", width=0)
        search_icon = self.load_image("Search.png")
        search_icon_label = tk.Label(canvas, image=search_icon, bg="#EFEFEF")
        search_icon_label.image = search_icon
        search_icon_label.place(relx=0.05, rely=0.5, anchor="w")

        self.search_entry = tk.Entry(canvas, font=self.font_raleway, bd=0, fg="gray", bg="#EFEFEF", justify="left", highlightthickness=0)
        self.search_entry.insert(0, "Search")
        self.search_entry.bind("<FocusIn>", self.on_search_focus_in)
        self.search_entry.bind("<FocusOut>", self.on_search_focus_out)
        self.search_entry.bind("<KeyRelease>", self.on_key_release)
        self.search_entry.place(x=50, rely=0.5, anchor="w", width=200, height=45)

        self.search_results_frame = tk.Frame(search_container, bg=self.content_bg_color)
        self.search_results_frame.pack(side="top", fill="x", padx=10)

        self.create_button(self.content_area, 150, 45, "View Analysis", "get.png", self.analyze_area, relx=0.5, y=520)

        self.rectangle_image = self.load_image("rectangle.png")
        self.black_rectangle = self.create_black_version(self.rectangle_image)
        self.rectangle_button = tk.Label(self.content_area, image=self.rectangle_image, bg=self.content_bg_color, cursor="hand2")
        self.rectangle_button.place(x=50, y=80)
        
        self.rectangle_button.bind("<Enter>", self.on_rectangle_hover)
        self.rectangle_button.bind("<Leave>", self.on_rectangle_leave)
        self.rectangle_button.bind("<Button-1>", self.start_drawing)

        upload_image = Image.open("images/upload.png")
        upload_photo = ImageTk.PhotoImage(upload_image)

        def select_image_for_analysis():
            self.file_path = filedialog.askopenfilename(title="Select Image for Analysis", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            if self.file_path:
                location_name = self.prompt_for_location_name()
                print(location_name)
                if location_name is None or location_name.strip() == "":
                    print("Cancelled", "No location name entered. Action cancelled.")
                    return
                self.location_name = location_name
                self.change_page(1)
                self.watchlist_checkbox.config(state="disabled")

        upload_button = tk.Label(self.content_area, image=upload_photo, bg=self.content_bg_color, cursor="hand2")
        upload_button.image = upload_photo
        upload_button.place(x=50, y=150)
        upload_button.bind("<Button-1>",lambda event: select_image_for_analysis())

    def show_contact_page(self):
        tk.Frame(self.content_area, bg=self.border_color, width=702, height=2).place(relx=0.5, y=20, anchor="n")
        tk.Label(self.content_area, text="Contact", font=self.font_raleway_bold, bg=self.content_bg_color, fg=self.control_panel_color).place(x=95, y=30)
        tk.Label(self.content_area, text="Receive your forest coverage analysis report", font=self.font_raleway, bg=self.content_bg_color, fg="#666666").place(x=95, y=80)
        tk.Label(self.content_area, text="Enter email:", font=self.font_rasa, bg=self.content_bg_color, fg="black").place(relx=0.5, y=250, anchor="s")

        self.setup_email_input()
        tk.Label(self.content_area, text="Enter contact number:", font=self.font_rasa, bg=self.content_bg_color, fg="black").place(relx=0.5, y=340, anchor="s")
        self.setup_contact_number_input()
        options_frame = tk.Frame(self.content_area, bg=self.content_bg_color)
        options_frame.place(relx=0.5, y=400, anchor="center")
        
        self.watchlist_var = tk.BooleanVar()
        self.watchlist_checkbox = tk.Checkbutton(
            options_frame, 
            text="Add to watchlist",
            variable=self.watchlist_var,
            font=self.font_dubai,
            bg=self.content_bg_color,
            fg="black",
            selectcolor=self.content_bg_color
        )
        self.watchlist_checkbox.pack(side="left", padx=(0, 10))
        
        self.timeframe_var = tk.StringVar(value="Weekly")
        self.timeframe_dropdown = tk.OptionMenu(
            options_frame,
            self.timeframe_var,
            "Daily",
            "Weekly",
            "Monthly"
        )
        self.timeframe_dropdown.config(
            font=self.font_dubai,
            bg=self.content_bg_color,
            highlightthickness=0,
            borderwidth=1
        )
        self.timeframe_dropdown["menu"].config(
            font=self.font_raleway,
            bg=self.content_bg_color
        )
        self.timeframe_dropdown.pack(side="left")
        
        self.create_button(self.content_area, 150, 45, "Confirm", "Tick.png", self.handle_email_submit, relx=0.5, y=430)

    def setup_contact_number_input(self):
        canvas = tk.Canvas(self.content_area, width=300, height=40, bg=self.content_bg_color, highlightthickness=0)
        canvas.place(relx=0.5, y=330, anchor="n")
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, 300, 40, 15, fill="#EFEFEF", width=0)

        self.contact_number_entry = tk.Entry(canvas, font=self.font_raleway, bd=0, fg="gray", bg="#EFEFEF", highlightthickness=0, width=30)
        self.contact_number_entry.insert(0, "123-456-7890")
        self.contact_number_entry.place(relx=0.5, rely=0.5, anchor="center")
        self.contact_number_entry.bind("<FocusIn>", self.on_contact_number_focus_in)
        self.contact_number_entry.bind("<FocusOut>", self.on_contact_number_focus_out)

    def on_contact_number_focus_in(self, event):
        if self.contact_number_entry.get() == "123-456-7890":
            self.contact_number_entry.delete(0, tk.END)
            self.contact_number_entry.config(fg="black")

    def on_contact_number_focus_out(self, event):
        if not self.contact_number_entry.get():
            self.contact_number_entry.insert(0, "123-456-7890")
            self.contact_number_entry.config(fg="gray")
    
    def show_ai_page(self, analysis_received=False):
        for widget in self.content_area.winfo_children():
            widget.destroy()

        tk.Frame(self.content_area, bg=self.border_color, width=702, height=2).place(relx=0.5, y=20, anchor="n")
        tk.Label(self.content_area, text="AI Analysis", font=self.font_raleway_bold, bg=self.content_bg_color, fg=self.control_panel_color).place(relx=0.5, y=30, anchor="n")

        if analysis_received:
            self.predicted_mask_image_label = tk.Label(self.content_area, bg=self.content_bg_color)
            self.predicted_mask_image_label.place(relx=0.5, y=80, anchor="n")
            self.display_predicted_mask()
        else:
            if hasattr(self, 'predicted_mask_image_label') and self.predicted_mask_image_label.winfo_exists():
                self.predicted_mask_image_label.place_forget()

        rectangle_height = 400 if not analysis_received else 250
        print(f"rectangle_height: {rectangle_height}")

        self.output_canvas = tk.Canvas(self.content_area, bg="#EFEFEF", width=650, height=rectangle_height, highlightthickness=0)

        if not analysis_received:
            self.output_canvas.place(relx=0.5, y=self.content_area.winfo_height() // 2 - rectangle_height // 2, anchor="n")
        else:
            self.output_canvas.place(relx=0.5, y=self.content_area.winfo_height() - 10, anchor="s")

        self.output_canvas.create_rounded_rectangle = self.create_rounded_rectangle(self.output_canvas)
        self.output_canvas.create_rounded_rectangle(0, 0, 650, rectangle_height, 15, fill="#EFEFEF", width=0)

        text_frame = tk.Frame(self.output_canvas, bg="#EFEFEF")
        self.output_canvas.create_window((0, 0), window=text_frame, anchor="nw")

        scrollbar = tk.Scrollbar(text_frame, width=4)
        scrollbar.pack(side="right", fill="y")

        self.output_text = tk.Text(text_frame, wrap=tk.WORD, width=70, height=10, font=self.font_raleway, bg="#EFEFEF", fg="black", bd=0, highlightthickness=0, yscrollcommand=scrollbar.set, padx=10, pady=10)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output_text.yview)

        if analysis_received:
            self.output_text.config(state="normal")
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", self.ai_analysis_text)
            self.output_text.config(state="disabled")

        if not analysis_received:
            self.output_text.config(state="normal")
            self.output_text.insert("1.0", "Waiting for AI analysis response...")
            self.output_text.config(state="disabled")

    def setup_email_input(self):
        canvas = tk.Canvas(self.content_area, width=300, height=40, bg=self.content_bg_color, highlightthickness=0)
        canvas.place(relx=0.5, y=240, anchor="n")
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, 300, 40, 15, fill="#EFEFEF", width=0)

        self.email_entry = tk.Entry(canvas, font=self.font_raleway, bd=0, fg="gray", bg="#EFEFEF", highlightthickness=0, width=30)
        self.email_entry.insert(0, "john@doe.com")
        self.email_entry.place(relx=0.5, rely=0.5, anchor="center")
        self.email_entry.bind("<FocusIn>", self.on_email_focus_in)
        self.email_entry.bind("<FocusOut>", self.on_email_focus_out)

    def setup_output_box(self):
        canvas = tk.Canvas(self.content_area, width=500, height=300, bg=self.content_bg_color, highlightthickness=0)
        canvas.place(relx=0.5, y=159, anchor="n")
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, 500, 300, 15, fill="#EFEFEF", width=0)

        self.output_text = tk.Text(canvas, font=self.font_raleway, bg="#EFEFEF", fg="black", bd=0, highlightthickness=0, width=48, height=11)
        self.output_text.place(relx=0.5, rely=0.5, anchor="center")
        self.output_text.insert("1.0", "AI analysis will appear here...")
        self.output_text.config(state="disabled")
    
    def safe_scroll(self, widget, event):
        try:
            widget.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except (tk.TclError, AttributeError):
            # Widget was destroyed or never existed
            pass
        
    def show_about_page(self):
        tk.Frame(self.content_area, bg=self.border_color, width=800, height=2).place(relx=0.5, y=20, anchor="n")
        tk.Label(self.content_area, text="ReLeaf was created by", font=self.font_raleway_bold, bg=self.content_bg_color, fg=self.control_panel_color).place(relx=0.5, y=30, anchor="n")

        scroll_canvas = tk.Canvas(self.content_area, bg=self.content_bg_color)
        scroll_canvas.place(relx=0.5, y=80, anchor="n", width=800, height=400)

        scrollbar = tk.Scrollbar(scroll_canvas, width=20, command=scroll_canvas.yview)
        scrollbar.pack(side="right", fill="y")

        self.scrollable_frame = tk.Frame(scroll_canvas, bg=self.content_bg_color)
        scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind("<Configure>", lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))
        scroll_canvas.configure(yscrollcommand=scrollbar.set)

        self.create_profile_box(0, "Tinkerer", "Ved", "Shivane", "Tinkerer and programmer, loves writing code to solve real world problems. Is keenly interested in AI and ML projects and is studying various advanced methods of implementing them, and applying them in projects like ReLeaf.", "ved.png", self.scrollable_frame)
        self.create_profile_box(390, "Innovator", "Vijay Ganesh", "Kumar", "I am a curious and creative 9th-grade student following the Cambridge curriculum. Passionate about learning, I excel in writing and enjoys exploring topics like public speaking, current events, and culture. I am a thoughtful, hardworking, and driven individual.", "vijay.png", self.scrollable_frame)

        self.create_profile_box(45, "Developer", "Keerthan", "Bhaskaran", "As a developer for ReLeaf, I integrate AI and geospatial data to combat deforestation, monitor forest changes, and support reforestation and sustainable land management efforts.", "keerthan.png", self.scrollable_frame)

        for text, y in [("© 2025-26 Ved, Vijay and Keerthan, All rights reserved", 520), ("Opensourced under the MIT license", 560)]:
            tk.Label(self.content_area, text=text, font=("Raleway", 10, "bold"), bg=self.content_bg_color, fg=self.control_panel_color).place(relx=0.5, y=y, anchor="center")

        self.content_area.bind_all("<MouseWheel>", lambda event: self.safe_scroll(scroll_canvas, event))

    def create_profile_box(self, x_pos, title, first_name, last_name, description, image_name, parent_frame):
        canvas = tk.Canvas(parent_frame, width=380, height=280, bg=self.content_bg_color, highlightthickness=0)
        canvas.pack(side="top", padx=200, pady=10)
        canvas.create_rounded_rectangle = self.create_rounded_rectangle(canvas)
        canvas.create_rounded_rectangle(0, 0, 380, 280, 15, fill="#EFEFEF", width=0)
        
        tk.Label(canvas, text=title, font=self.font_raleway, bg="#EFEFEF", fg="#777777").place(x=20, y=20)
        tk.Label(canvas, text=first_name, font=("Raleway", 24), bg="#EFEFEF", fg=self.control_panel_color).place(x=20, y=50)
        tk.Label(canvas, text=last_name, font=("Raleway", 16), bg="#EFEFEF", fg=self.control_panel_color).place(x=20, y=85)

        profile_image = self.load_image(image_name)
        image_label = tk.Label(canvas, image=profile_image, bg="#EFEFEF")
        image_label.image = profile_image
        image_label.place(x=220, y=-20)

        tk.Frame(canvas, bg="white", width=113, height=2).place(x=20, y=120)
        tk.Label(canvas, text=description, font=self.font_raleway, bg="#EFEFEF", fg="#666666", wraplength=340, justify="left").place(x=20, y=140)

    def on_key_release(self, event):
        address = self.search_entry.get()
        current_time = time.time()
        
        if current_time - self.last_search_time >= self.debounce_delay and address and not self.is_searching:
            self.last_search_time = current_time
            self.is_searching = True
            self.clear_search_results()

            if self.search_thread and self.search_thread.is_alive():
                self.search_thread.join()

            self.search_thread = threading.Thread(target=self.show_address_preview, args=(address,))
            self.search_thread.daemon = True
            self.search_thread.start()

    def search_address(self, address): 
        response = requests.get(f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={self.GOOGLE_API_KEY}")
        return response.json()["results"] if response.status_code == 200 else []

    def show_address_preview(self, address):
        for suggestion in self.search_address(address):
            location = suggestion['geometry']['location']
            self.root.after(0, self.add_search_result, suggestion['formatted_address'], float(location["lat"]), float(location["lng"]))
        self.is_searching = False

    def add_search_result(self, display_name, lat, lon):
        result_label = tk.Label(self.search_results_frame, text=display_name, fg="blue", cursor="hand2", bg=self.content_bg_color, font=self.font_raleway)
        result_label.pack(anchor="w", pady=2)
        result_label.bind("<Button-1>", lambda e: self.select_address(lat, lon, display_name))

    def select_address(self, lat, lon, display_name):
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, display_name)
        self.map_view.set_position(lat, lon)
        self.map_view.set_zoom(15)
        
        self.map_view.delete_all_marker()
        
        self.map_view.set_marker(lat, lon)
        
        self.clear_search_results()

    def clear_search_results(self): 
        for widget in self.search_results_frame.winfo_children(): widget.destroy()

    def create_black_version(self, photo_image):
        pil_image = Image.open("images/rectangle.png")
        black_image = Image.new('RGBA', pil_image.size, 'black')
        black_image.putalpha(pil_image.getchannel('A'))
        return ImageTk.PhotoImage(black_image)

    def handle_server_response(self, response_data):
        def process_response():
            try:
                self.change_page(2)
                self.output_text.config(state="normal")
                self.output_text.delete("1.0", tk.END)
                self.output_text.insert("1.0", "Processing analysis...")
                self.output_text.config(state="disabled")
                self.root.update()

                image_data = base64.b64decode(response_data["predicted_mask_base64"])
                
                with open("predicted_mask.png", "wb") as f:
                    f.write(image_data)
                    
                self.ai_analysis_text = response_data.get('ai_analysis', 'AI analysis not available')
                self.display_ai_analysis(self.ai_analysis_text)
                
                print("Calling show_ai_page with analysis_received=True")
                self.show_ai_page(analysis_received=True)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to process server response: {str(e)}")

        threading.Thread(target=process_response, daemon=True).start()

    def display_ai_analysis(self, analysis_text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", analysis_text)
        self.output_text.config(state="disabled")

    def handle_email_submit(self):
        if not hasattr(self, 'min_lat') and self.file_path == "":
            messagebox.showerror("Error", "Please select an area on the map first.")
            return

        email = self.email_entry.get()
        number = self.contact_number_entry.get()
        if email == "john@doe.com":
            email = ""
        if number == "123-456-7890":
            number = ""
        email_pattern = r"[^@]+@[^@]+\.[^@]+"
        if not re.match(email_pattern, email):
            messagebox.showerror("Error", "Please enter a valid email address.")
            return

        if number.strip() != '':
            number_pattern = r"^\+?\d{7,15}$" 
            if not re.match(number_pattern, number):
                messagebox.showerror("Error", "Please enter a valid contact number (e.g., +1234567890).")
                return

        if self.current_rectangle:
            watchlist = self.watchlist_var.get()
            timeframe = self.timeframe_var.get()
            min_lat = self.min_lat
            max_lat = self.max_lat
            min_lon = self.min_lon
            max_lon = self.max_lon
            current_zoom = self.current_zoom

        self.change_page(2)
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "Analyzing selected area...\nThis may take a few moments.")
        self.output_text.config(state="disabled")
        self.root.update()

        def submit_request():
            try:
                if not hasattr(self, "file_path"):
                    self.file_path = ""
                if self.current_rectangle and (self.file_path is None or self.file_path == ''):
                    data = {
                        "email": email,
                        "number": number.strip(),  # Send as empty string if not provided
                        "watchlist": watchlist,
                        "timeframe": timeframe,
                        "min_lat": min_lat,
                        "max_lat": max_lat,
                        "min_long": min_lon,
                        "max_long": max_lon,
                        "zoom": current_zoom
                    }
                else:
                    with open(self.file_path, "rb") as image_file:
                        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
                    data = {
                        "email": email,
                        "number": number.strip(),
                        "image": encoded_image,
                        "location": self.location_name
                    }

                self.file_path = None
                response = requests.post(
                    'https://api.projectreleaf.xyz/predict',
                    json=data
                )

                if response.status_code == 200:
                    self.root.after(0, lambda: self.handle_server_response(response.json()))
                    self.root.after(0, lambda: messagebox.showinfo("Success", "Your request has been submitted successfully!"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Server returned status code: {response.status_code}"))

            except requests.exceptions.RequestException as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to connect to server: {str(e)}"))


        threading.Thread(target=submit_request, daemon=True).start()

    def on_rectangle_hover(self, event):
        self.rectangle_button.configure(image=self.black_rectangle)

    def on_rectangle_leave(self, event):
        self.rectangle_button.configure(image=self.rectangle_image)
    
    def on_search_focus_in(self, event):
        if self.search_entry.get() == "Search": 
            self.search_entry.delete(0, tk.END)
            self.search_entry.config(fg="black")

    def on_search_focus_out(self, event):
        if not self.search_entry.get():
            self.search_entry.insert(0, "Search")
            self.search_entry.config(fg="black")

    def on_email_focus_in(self, event):
        if self.email_entry.get() == "john@doe.com":
            self.email_entry.delete(0, tk.END)
            self.email_entry.config(fg="black")

    def on_email_focus_out(self, event):
        if not self.email_entry.get():
            self.email_entry.insert(0, "john@doe.com")
            self.email_entry.config(fg="gray")

    def get_ai_analysis(self):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", "Waiting for AI analysis response...")
        self.output_text.config(state="disabled")

    def on_map_click(self, coordinates):
        if self.drawing:
            lat, lon = coordinates
            self.markers.append(self.map_view.set_marker(lat, lon, text="Point A"))
            if not self.start_coords:
                self.start_coords = coordinates
            else:
                self.end_coords = coordinates
                self.draw_rectangle_on_map(self.start_coords, self.end_coords)
                self.drawing = False
                self.rectangle_button.configure(image=self.rectangle_image)

    def draw_rectangle_on_map(self, start_coords, end_coords):
        lat1, lon1 = start_coords
        lat2, lon2 = end_coords
        self.current_rectangle = self.map_view.set_polygon([(lat1, lon1), (lat1, lon2), (lat2, lon2), (lat2, lon1)], fill_color="blue", outline_color="blue", border_width=2)

    def analyze_area(self):
        try:  
            if self.current_rectangle or self.file_path:
                try:
                    if self.file_path:
                        location_name = self.prompt_for_location_name()
                        if not location_name:
                            messagebox.showerror("Error", "Please enter a valid location name.")
                            return
                except:
                    pass
                try:
                    if self.current_rectangle:
                        self.export_image()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to export image: {str(e)}")
                self.change_page(1)

            else:
                messagebox.showwarning("No Selection", "Please select an area on the map first.")
        except:
            messagebox.showwarning("No Selection", "Please select an area on the map first.")
            return
    def prompt_for_location_name(self):
        while True:
            location_name = simpledialog.askstring("Location Name", "Enter the name of the location:")
            if location_name is None:
                return None
            location_name = location_name.strip()
            if len(location_name) >= 3:
                return location_name
            messagebox.showwarning("Invalid Input", "Please enter a meaningful location name (at least 3 characters).")

    def export_image(self):
        if not self.start_coords or not self.end_coords:
            messagebox.showerror("Error", "No rectangle selected to export.")
            return

        self.min_lat, self.max_lat = sorted([self.start_coords[0], self.end_coords[0]])
        self.min_lon, self.max_lon = sorted([self.start_coords[1], self.end_coords[1]])
        self.current_zoom = round(self.map_view.zoom)
        
        zoom = self.current_zoom
        tile_size = 256

    def start_drawing(self, event):
        try:
            self.watchlist_checkbox.config(state='active')
        except:
            pass
        self.drawing = True
        self.start_coords = self.end_coords = None
        if self.current_rectangle:
            self.map_view.delete(self.current_rectangle)
            self.current_rectangle = None
        if self.markers:
            self.map_view.delete_all_marker()
            self.markers.clear()
        self.rectangle_button.configure(image=self.black_rectangle)

    def display_predicted_mask(self):
        original_image = PILImage.open("predicted_mask.png")
        original_image.thumbnail((400, 400), Image.LANCZOS)

        self.predicted_mask_image = ImageTk.PhotoImage(original_image)

        self.predicted_mask_image_label.config(image=self.predicted_mask_image)
        self.predicted_mask_image_label.image = self.predicted_mask_image

if __name__ == "__main__":
    root = tk.Tk()
    app = ReLeafApp(root)
    root.mainloop()
