import tkinter as tk
from tkinter import filedialog
import json
import re
from nltk import word_tokenize
from nltk.tokenize import WhitespaceTokenizer

class LabelingApp:
    def __init__(self):

        self.total_items = []
        self.current_index = 0
        self.label_text = ['alloy2pum', 'pro2pum', 'prc2rum', 'alloy2prc', 'prc2pum', 'None']

        self.window = tk.Tk()
        self.window.title("Relation Labeling Tool")

        # Button to choose dataset file
        self.choose_button = tk.Button(self.window, text="Choose Dataset", command=self.choose_dataset)
        self.choose_button.pack()

        self.text_display = tk.Text(self.window, wrap="word", height=8, width=100, font=10)
        self.text_display.pack()

        self.label_var = tk.StringVar(value="")  # Variable to store the selected label
        self.label_options = ["Yes", "None"]

        self.info_label = tk.Label(self.window, text="")
        self.info_label.pack(pady=5)

        # Frame to hold label buttons with padding
        self.button_frame = tk.Frame(self.window)
        self.button_frame.pack(pady=10)  # Adjust the padding as needed

        self.label_buttons = []
        for option in self.label_options:
            button = tk.Button(self.button_frame, text=option, command=lambda o=option: self.mark_relation(o))
            button.pack(side=tk.LEFT, padx=30)
            self.label_buttons.append(button)

        self.status_label = tk.Label(self.window, text="")
        self.status_label.pack()

        # self.load_next_item()

        self.done_button = tk.Button(self.window, text="Done", command=self.done_labeling)
        self.done_button.pack()

    def choose_dataset(self):
        filename = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if filename:
            self.filename = filename
            data_lines = load_data(filename)
            if data_lines:
                self.total_items = data_lines
                self.load_next_item()

    def mark_relation(self, option):
        self.label_var.set(option)

    def load_next_item(self):
        if self.current_index < len(self.total_items):
            item = json.loads(self.total_items[self.current_index].strip())
            self.highlight_text(item)
            self.status_label.config(text=f"Row {self.current_index+1}/{len(self.total_items)}")
        else:
            self.status_label.config(text=f"All items labeled.")

    def preprocess_text(self, name):
        word = WhitespaceTokenizer()
        word_list = word.tokenize(name)
        # word_list = word_tokenize(name)
        for i in range(len(word_list)):
            if re.search('\\W', word_list[i][0]):
                word_list[i] = '\\'+word_list[i]
        name = ' '.join(word_list)
        return name

    def find_span(self, name, pos, text):
        word = WhitespaceTokenizer()
        word_list = word.tokenize(text)
        for idx, token in enumerate(word_list):
            if pos[0] == idx:
                start_id = len(' '.join(word_list[:idx]))
            if pos[1]-1 == idx:
                start = len(' '.join(word_list[:idx]))
                end_id = start + len(token)+1
        return start_id, end_id

    def highlight_text(self, item):
        h_name = item["h"]["name"]
        t_name = item["t"]["name"]
        h_name = self.preprocess_text(h_name)
        t_name = self.preprocess_text(t_name)
        relation = item["relation"]
        if relation == self.label_text[0]:
            bro_relation = "成分2性能数值"
        elif relation == self.label_text[1]:
            bro_relation = "性能2性能数值"
        elif relation == self.label_text[2]:
            bro_relation = "工艺2工艺数值"
        elif relation == self.label_text[3]:
            bro_relation = "成分2工艺"
        elif relation == self.label_text[4]:
            bro_relation = "工艺2性能数值"
        elif relation == self.label_text[-1]:
            bro_relation = "没关系"
        text_str = item["text"]
        text = text_str + '\talloy:red, pro:pink, pum:yellow, prc:green, rum:orange'+'\n\n'+'relation probably is '+bro_relation
        # match_obj1 = re.search(h_name+' ', text)
        # match_obj2 = re.search(t_name+' ', text)
        match_obj1=self.find_span(h_name, item["h"]["pos"], text_str)
        match_obj2 = self.find_span(t_name, item["t"]["pos"], text_str)
        h_pos = match_obj1
        t_pos = match_obj2
        self.text_display.delete(1.0, tk.END)
        self.text_display.insert(tk.END, text)
        if match_obj1 and match_obj2:
            print(match_obj1 , match_obj2)
            # h_pos = match_obj1.span()
            # t_pos = match_obj2.span()
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

            if relation == self.label_text[0]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="red")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="yellow")
            if relation == self.label_text[1]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="pink")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="yellow")
            if relation == self.label_text[2]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="green")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="orange")
            if relation == self.label_text[3]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="red")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="green")
            if relation == self.label_text[4]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="green")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="yellow")
            if relation == self.label_text[-1]:
                self.text_display.tag_add("h_entity", f"1.{h_pos[0]}", f"1.{h_pos[1]}")
                self.text_display.tag_config("h_entity", background="gray")
                self.text_display.tag_add("t_entity", f"1.{t_pos[0]}", f"1.{t_pos[1]}")
                self.text_display.tag_config("t_entity", background="gray")

    def done_labeling(self):
        relation_label = self.label_var.get()
        if relation_label:  # Check if a label is selected
            item = json.loads(self.total_items[self.current_index].strip())
            if relation_label=="None":
                item["relation"] = relation_label
            else:
                item = item
            self.current_index += 1
            self.load_next_item()

            self.save_data(item)
            self.label_var.set("")
        else:
            self.info_label.config(text=f"Please select a label before saving.")
            # messagebox.showwarning("Warning", "Please select a label before saving.")

    def save_data(self,item):
        try:
            with open(self.filename[:-5]+"_labeled.json", "a", encoding='utf-8') as f:
                json.dump(item, f)
                f.write("\n")
            path = self.filename[:-5]+"_labeled"
            label=item["relation"]
            if hasattr(self, "info_label") and self.info_label.winfo_exists():
                self.info_label.config(text=f"{label}    successfully saved as:\n{path}")
        except Exception as e:
            error_message = f"An error occurred while saving the data: {str(e)}"
            if hasattr(self, "info_label") and self.info_label.winfo_exists():
                self.info_label.config(text=error_message)
            else:
                print(error_message)  # Print the error message to console if label widget is not available
        # finally:
        #     self.window.destroy()

def load_data(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data_lines = f.readlines()
        return data_lines
    # return data_lines
    # try:
    #     with open(filename, "r",encoding='utf-8') as f:
    #         data_lines = f.readlines()
    #     return data_lines
    # except FileNotFoundError:
    #     messagebox.showerror("Error", f"File '{filename}' not found.")
    #     return []
    # except Exception as e:
    #     messagebox.showerror("Error", f"An error occurred while loading the data: {str(e)}")
    #     return []

def main():
    # filename = "data.json"  # Replace with the path to your dataset JSON file
    # data_lines = load_data(filename)
    app = LabelingApp()
    app.window.mainloop()

if __name__ == "__main__":
    main()
