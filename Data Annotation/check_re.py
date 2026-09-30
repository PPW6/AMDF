import tkinter as tk
import json
import re
from nltk.tokenize import WhitespaceTokenizer
from tkinter import filedialog

class LabelingApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Relation Highlighting Tool")

        self.text_display = tk.Text(self.window, wrap="word", height=8, width=100, font=("Helvetica", 12))
        self.text_display.pack()

        self.status_label = tk.Label(self.window, text="")
        self.status_label.pack()

        self.load_button = tk.Button(self.window, text="Load Data", command=self.load_data)
        self.load_button.pack()

        self.navigation_frame = tk.Frame(self.window)
        self.navigation_frame.pack(pady=10)

        self.prev_button = tk.Button(self.navigation_frame, text="Previous", command=self.load_previous_item)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(self.navigation_frame, text="Next", command=self.load_next_item)
        self.next_button.pack(side=tk.LEFT, padx=5)

    def load_data(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            with open(filename, "r", encoding='utf-8') as f:
                self.total_items = f.readlines()
                self.current_index = 0
                self.load_current_item()

    def preprocess_text(self, name):
        word = WhitespaceTokenizer()
        word_list = word.tokenize(name)
        for i in range(len(word_list)):
            if re.search('\\W', word_list[i][0]):
                word_list[i] = '\\' + word_list[i]
        name = ' '.join(word_list)
        return name

    def find_span(self, name, pos, text):
        word = WhitespaceTokenizer()
        word_list = word.tokenize(text)
        start_id, end_id = 0, 0
        for idx, token in enumerate(word_list):
            if pos[0] == idx:
                start_id = len(' '.join(word_list[:idx]))
            if pos[1] - 1 == idx:
                start = len(' '.join(word_list[:idx]))
                end_id = start + len(token) + 1
        return start_id, end_id

    def load_current_item(self):
        if 0 <= self.current_index < len(self.total_items):
            item = json.loads(self.total_items[self.current_index].strip())
            self.highlight_text(item)
            self.status_label.config(text=f"Row {self.current_index + 1}/{len(self.total_items)}")
        else:
            self.status_label.config(text="No more items.")

    def load_next_item(self):
        if self.current_index < len(self.total_items) - 1:
            self.current_index += 1
            self.load_current_item()

    def load_previous_item(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_item()

    def highlight_text(self, item):
        h_name = item["h"]["name"]
        t_name = item["t"]["name"]
        h_name = self.preprocess_text(h_name)
        t_name = self.preprocess_text(t_name)
        relation = item["relation"]
        relation_label = {
            "alloy2pum": "成分2性能数值",
            "pro2pum": "性能2性能数值",
            "prc2rum": "工艺2工艺数值",
            "alloy2prc": '成分2工艺',
            "prc2pum": '工艺2性能数值',
            "None": "没关系"
        }.get(relation, relation)
        text_str = item["text"]
        text = text_str + '\talloy:red, pro:pink, pum:yellow, prc:green, rum:orange' + '\n\n' + 'relation probably is ' + relation_label
        match_obj1 = self.find_span(h_name, item["h"]["pos"], text_str)
        match_obj2 = self.find_span(t_name, item["t"]["pos"], text_str)
        h_pos = match_obj1
        t_pos = match_obj2
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, text)
        if match_obj1 and match_obj2:
            match_red = re.search(':red', text)
            match_pink = re.search(':pink', text)
            match_yellow = re.search(':yellow', text)
            match_green = re.search(':green', text)
            match_orange = re.search(':orange', text)
            r_pos = match_red.span()
            p_pos = match_pink.span()
            y_pos = match_yellow.span()
            g_pos = match_green.span()
            o_pos = match_orange.span()

            self.text_display.tag_add("r_entity", f"1.{r_pos[0]}", f"1.{r_pos[1]}")
            self.text_display.tag_config("r_entity", background="red")
            self.text_display.tag_add("p_entity", f"1.{p_pos[0]}", f"1.{p_pos[1]}")
            self.text_display.tag_config("p_entity", background="pink")
            self.text_display.tag_add("y_entity", f"1.{y_pos[0]}", f"1.{y_pos[1]}")
            self.text_display.tag_config("y_entity", background="yellow")
            self.text_display.tag_add("g_entity", f"1.{g_pos[0]}", f"1.{g_pos[1]}")
            self.text_display.tag_config("g_entity", background="green")
            self.text_display.tag_add("o_entity", f"1.{o_pos[0]}", f"1.{o_pos[1]}")
            self.text_display.tag_config("o_entity", background="orange")

            relation_colors = {
                "alloy2pum": ("red", "yellow"),
                "pro2pum": ("pink", "yellow"),
                "prc2rum": ("green", "orange"),
                "alloy2prc": ("red", "green"),
                "prc2pum": ("green", "yellow"),
                "None": ("gray", "gray")
            }
            h_color, t_color = relation_colors.get(relation, ("gray", "gray"))
            self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
            self.text_display.tag_config("h_entity", background=h_color)
            self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
            self.text_display.tag_config("t_entity", background=t_color)

def main():
    app = LabelingApp()
    app.window.mainloop()

if __name__ == "__main__":
    main()

