import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from datetime import datetime, timedelta

class TaskManager:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        self.add_patterns()
        self.tasks = []

    def add_patterns(self):
        task_pattern = [{"POS": "VERB"}, {"POS": "NOUN"}]
        due_date_pattern = [{"LOWER": "due"}, {"LOWER": {"IN": ["on", "by"]}}, {"ENT_TYPE": "DATE"}]
        priority_pattern = [{"LOWER": "priority"}, {"LOWER": "is"}, {"LOWER": {"IN": ["high", "medium", "low"]}}]
        category_pattern = [{"LOWER": "category"}, {"LOWER": "is"}, {"POS": "NOUN"}]
        assignee_pattern = [{"LOWER": "assignee"}, {"LOWER": "is"}, {"POS": "PROPN"}]

        self.matcher.add("TASK", [task_pattern])
        self.matcher.add("DUE_DATE", [due_date_pattern])
        self.matcher.add("PRIORITY", [priority_pattern])
        self.matcher.add("CATEGORY", [category_pattern])
        self.matcher.add("ASSIGNEE", [assignee_pattern])

    def extract_tasks(self, text):
        doc = self.nlp(text)
        matches = self.matcher(doc)
        current_task = {}

        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]

            if label == "TASK":
                if current_task:
                    self.tasks.append(current_task)
                current_task = {"task": span.text, "status": "pending"}
            elif label == "DUE_DATE":
                due_date = doc[end-1].text
                current_task["due_date"] = self.parse_date(due_date)
            elif label == "PRIORITY":
                priority = doc[end-1].text.lower()
                current_task["priority"] = priority
            elif label == "CATEGORY":
                category = doc[end-1].text
                current_task["category"] = category
            elif label == "ASSIGNEE":
                assignee = doc[end-1].text
                current_task["assignee"] = assignee

        if current_task:
            self.tasks.append(current_task)

        return self.tasks

    def parse_date(self, date_string):
        today = datetime.now().date()
        if date_string.lower() == "today":
            return today
        elif date_string.lower() == "tomorrow":
            return today + timedelta(days=1)
        elif date_string.lower().startswith("next"):
            days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6}
            day = date_string.lower().split()[1]
            days_ahead = days[day] - today.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            return today + timedelta(days=days_ahead)
        else:
            try:
                return datetime.strptime(date_string, "%Y-%m-%d").date()
            except ValueError:
                return None

    def update_task_status(self, task_id, status):
        if 0 <= task_id < len(self.tasks):
            self.tasks[task_id]["status"] = status
            return self.tasks[task_id]
        return None

    def filter_tasks(self, **kwargs):
        filtered_tasks = self.tasks
        for key, value in kwargs.items():
            filtered_tasks = [task for task in filtered_tasks if task.get(key) == value]
        return filtered_tasks

    def sort_tasks_by_priority(self):
        priority_order = {"high": 3, "medium": 2, "low": 1}
        return sorted(self.tasks, key=lambda x: priority_order.get(x.get("priority", "low"), 0), reverse=True)

    def sort_tasks_by_due_date(self):
        return sorted(self.tasks, key=lambda x: x.get("due_date") or datetime.max.date())

    def generate_task_summary(self):
        total_tasks = len(self.tasks)
        pending_tasks = len(self.filter_tasks(status="pending"))
        in_progress_tasks = len(self.filter_tasks(status="in progress"))
        completed_tasks = len(self.filter_tasks(status="completed"))

        summary = f"Task Summary:\n"
        summary += f"Total Tasks: {total_tasks}\n"
        summary += f"Pending Tasks: {pending_tasks}\n"
        summary += f"In Progress Tasks: {in_progress_tasks}\n"
        summary += f"Completed Tasks: {completed_tasks}\n"

        return summary

    def get_upcoming_tasks(self, days=7):
        today = datetime.now().date()
        upcoming_date = today + timedelta(days=days)
        upcoming_tasks = [task for task in self.tasks if task.get("due_date") and today <= task["due_date"] <= upcoming_date]
        return sorted(upcoming_tasks, key=lambda x: x["due_date"])

    def get_overdue_tasks(self):
        today = datetime.now().date()
        overdue_tasks = [task for task in self.tasks if task.get("due_date") and task["due_date"] < today and task["status"] != "completed"]
        return sorted(overdue_tasks, key=lambda x: x["due_date"])

    def add_task(self, task_description):
        new_tasks = self.extract_tasks(task_description)
        return new_tasks[-1] if new_tasks else None

    def delete_task(self, task_id):
        if 0 <= task_id < len(self.tasks):
            return self.tasks.pop(task_id)
        return None

    def clear_completed_tasks(self):
        completed_tasks = self.filter_tasks(status="completed")
        self.tasks = [task for task in self.tasks if task["status"] != "completed"]
        return completed_tasks