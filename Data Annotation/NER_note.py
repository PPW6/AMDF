
import tkinter as tk
from tkinter import filedialog


class DataAnnotator:
    def __init__(self, root):
        self.root = root
        self.dataset_path = None
        self.label_path = None
        self.labels = []
        self.labels_defin = []
        self.annotated_data = {}
        self.current_page = 1
        self.items_per_page = 15
        self.num_pages = 1
        self.done_flag = False

        # Create the main window
        self.root.title("Data Annotation Tool")

        # Create a button to browse for the dataset file
        self.browse_dataset_button = tk.Button(root, text="Browse Dataset File", command=self.browse_dataset)
        self.browse_dataset_button.pack(pady=10)

        # Create a button to browse for the label file
        self.browse_label_button = tk.Button(root, text="Browse Label File", command=self.browse_label)
        self.browse_label_button.pack(pady=5)

        # Create a text widget with a border
        self.text_with_box = tk.Text(root, height=10, width=170, wrap=tk.WORD, bd=2, relief=tk.SOLID)
        self.text_with_box.pack(pady=5)

        # Create a label to display lable definition
        self.defin_label = tk.Label(root, text="")
        self.defin_label.pack(pady=5)

        # Create a frame to display the annotated data
        self.data_frame = tk.Frame(root)
        self.data_frame.pack(pady=5)

        # Create navigation buttons for page change
        self.prev_page_button = tk.Button(root, text="Prev", command=self.prev_page, state=tk.DISABLED)
        self.prev_page_button.pack(side=tk.LEFT, padx=10)
        self.next_page_button = tk.Button(root, text="Next", command=self.next_page, state=tk.DISABLED)
        self.next_page_button.pack(side=tk.RIGHT, padx=10)

        # Create a label to display information
        self.info_label = tk.Label(root, text="")
        self.info_label.pack(pady=5)

        # Create a "Done" button to save the annotated file
        self.done_button = tk.Button(root, text="Done", command=self.save_annotated_file, state=tk.DISABLED)
        self.done_button.pack(pady=5)



    def browse_dataset(self):
        self.dataset_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.dataset_path:
            self.load_dataset()

    def browse_label(self):
        self.label_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if self.label_path:
            self.load_labels()

    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            self.num_pages = len(lines) // self.items_per_page + (1 if len(lines) % self.items_per_page != 0 else 0)
            self.annotated_data = {i + 1: {'text': line.strip(), 'label': None} for i, line in enumerate(lines)}
            self.display_current_page()

        self.info_label.config(text=f"Dataset file loaded: {self.dataset_path}")

    def load_labels(self):
        with open(self.label_path, 'r', encoding='utf-8') as file:
            labels_all = [line.strip() for line in file.readlines()]
            self.labels = labels_all[:-1]
            self.labels_defin = labels_all[-1]
        self.info_label.config(text=f"Label file loaded: {self.label_path}")
        self.display_current_page()

    def display_current_page(self):
        for widget in self.data_frame.winfo_children():
            widget.destroy()
        # Display items for the current page
        start_index = (self.current_page - 1) * self.items_per_page
        end_index = min(start_index + self.items_per_page, len(self.annotated_data))
        children_count = len(self.data_frame.winfo_children())

        for i in range(start_index, end_index):
            if i + 1 in self.annotated_data:
                label_frame = tk.Frame(self.data_frame)
                label_frame.pack(anchor='w')

                num = i + 1
                label_text = tk.Label(label_frame, text=f"{num} {self.annotated_data[i + 1]['text']}")
                label_text.pack(side='left', padx=10)

                for label in self.labels:
                    label_button = tk.Button(label_frame, text=label,
                                             command=lambda l=label, index=i + 1: self.annotate_data(l, index))
                    label_button.pack(side='left', padx=5)

        # Update navigation buttons
        self.prev_page_button.config(state=tk.NORMAL if self.current_page > 1 else tk.DISABLED)
        self.next_page_button.config(state=tk.NORMAL if self.current_page < self.num_pages else tk.DISABLED)
        self.done_button.config(state=tk.NORMAL )#if self.done_flag else tk.DISABLED



    def prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.display_current_page()

    def next_page(self):
        if self.current_page < self.num_pages:
            self.current_page += 1
            self.display_current_page()

    def annotate_data(self, label, index):
        self.annotated_data[index]['label'] = label
        self.done_flag = all(item['label'] is not None for item in self.annotated_data.values())
        # Display the annotated label with the data
        annotated_text = f"{index,self.annotated_data[index]['text']} - {label}"
        self.defin_label.config(text=f"labels_definition: {self.labels_defin}")
        self.info_label.config(text=f"Annotated: {annotated_text}")
        # self.info_label.config(text=f"Annotated: {self.annotated_data.items()}")
        # Insert text into the widget
        self.text_with_box.delete(1.0, tk.END) #clear previous data
        data_strings = [str(index)+':'+str(item['text'])+' '+str(item['label']) for index,item in self.annotated_data.items()]
        for data_string in data_strings[9::10]:
            id = int(data_string.split(':')[0])-1
            data_strings[id] = data_strings[id]+'\n'
        data_strings = ' '.join(data_strings)
        # text = [item['text'] for index, item in self.annotated_data.items()]
        # text = ' '.join(text)
        # Update the text of the button corresponding to the annotated data
        self.update_button(index)

        # Update the text widget
        self.update_text_widget()

        # Update the navigation buttons
        self.prev_page_button.config(state=tk.NORMAL if self.current_page > 1 else tk.DISABLED)
        self.next_page_button.config(state=tk.NORMAL if self.current_page < self.num_pages else tk.DISABLED)
        # self.done_button.config(state=tk.NORMAL if self.done_flag else tk.DISABLED)
        # self.done_button.config(state=tk.NORMAL )
    def update_button(self, index):
        # Get the button corresponding to the annotated data
        for widget in self.data_frame.winfo_children():
            children = widget.winfo_children()
            if children and children[0].cget("text") == str(index):
                button_to_update = children[1]  # Assuming the button is the second widget
                button_to_update.config(text=self.annotated_data[index]['label'])
                break

    def update_text_widget(self):
        self.text_with_box.delete(1.0, tk.END)  # clear previous data
        data_strings = [f"{index}: {item['text']} {item['label']}" for index, item in self.annotated_data.items()]
        for data_string in data_strings[9::10]:
            id = int(data_string.split(':')[0]) - 1
            data_strings[id] += '\n'
        data_text = ' '.join(data_strings)
        self.text_with_box.insert(tk.END, data_text)

    def save_annotated_file(self):
        if self.done_flag:
            if self.dataset_path:
                annotated_file_path = self.dataset_path.replace('.txt', '_annotated.txt')
                with open(annotated_file_path, 'w', encoding='utf-8') as file:
                    for line_num, item in self.annotated_data.items():
                        if len(item['text']) != 0:
                            file.write(f"{item['text']} {item['label']}\n")
                        else:
                            file.write(f"\n")
                self.info_label.config(text=f"Annotated file saved as:\n{annotated_file_path}")
            else:
                self.info_label.config(text="Error: No dataset file loaded.")
        else:
            # Find unlabeled data
            unlabeled_rows = [index for index, item in self.annotated_data.items() if item['label'] is None]
            if unlabeled_rows:
                self.info_label.config(
                    text=f"Error: Not all data annotated. Unlabeled rows: {', '.join(map(str, unlabeled_rows))}")
            else:
                self.info_label.config(text="Error: Not all data annotated.")


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x830")
    DataAnnotator(root)
    root.mainloop()