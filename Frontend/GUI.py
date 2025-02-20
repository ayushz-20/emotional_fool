from PyQt5.QtWidgets import QApplication, QMainWindow , QTextEdit , QStackedWidget , QWidget , QLineEdit , QGridLayout , QVBoxLayout , QHBoxLayout , QSizePolicy , QLabel , QPushButton, QFrame
from PyQt5.QtGui import QFont , QIcon , QPainter , QMovie , QColor, QTextCharFormat , QPixmap , QTextBlockFormat
from PyQt5.QtCore import Qt , QSize , QTime , QTimer
import sys
from dotenv import dotenv_values
import os


env_vars = dotenv_values(".env")
Assistantname = env_vars.get("Assistantname")
current_dir = os.getcwd()
old_chat_message = ""
TempDirPath = rf"{current_dir}\Frontend\Files"
GraphicsDirPath = rf"{current_dir}\Frontend\Graphics"

def AnswerModifier(Answer):
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    modified_answer = '/n'.join(non_empty_lines)
    return modified_answer

def QueryModifier(query):
    new_query = query.lower().strip()
    query_words = new_query.split()
    question_words = ["how","what","who","where","why","which","whose","whom","can you", "what's", "where's","how's"]
    
    
    if any(word + " " in new_query for word in question_words):
        if query_words[-1][-1] in ['.', '?', '!']:
            new_query = new_query[:-1] + "?"
        else:
            new_query += "?"
            
    else:
        if query_words[-1][-1] in ['.', '?', '!']:
            new_query = new_query[:-1] + "."
        else:
            new_query += "."
            
    return new_query.capitalize()

def SetMicrophoneStatus(command):
    with open(rf'{TempDirPath}\mic.data' , "w", encoding='utf-8') as file:
        file.write(command)
        
def GetMicrophoneStatus():
    with open(rf'{TempDirPath}\mic.data', "r", encoding='utf-8') as file:
        Status = file.read()
        
    return Status

def SetAssistantStatus(Status):
    with open(rf'{TempDirPath}\Status.data', "w", encoding='utf-8') as file:
        file.write(Status)
        
def GetAssistantStatus():
    with open(rf'{TempDirPath}\Status.data', "r", encoding='utf-8') as file :
        Status = file.read()
    return Status

def MicButtonInitialed():
    SetMicrophoneStatus("False")
def MicButtonClosed():
    SetMicrophoneStatus("True")
    
def GraphicsDirectoryPath(Filename):
    path = rf'{GraphicsDirPath}\{Filename}'
    return path

def TempDirectoryPath(Filename):
    path = rf'{TempDirPath}\{Filename}'
    return path

def ShowTextToScreen(Text):
    with open(rf'{TempDirPath}\Responses.data', "w" , encoding='utf-8') as file:
        file.write(Text)
        
        
class ChatSection(QWidget):
    def __init__(self):
        super(ChatSection, self).__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Adjust margins to positive values
        layout.setSpacing(10)  # Adjust spacing to a standard positive value

        self.chat_text_edit = QTextEdit()
        self.chat_text_edit.setReadOnly(True)
        self.chat_text_edit.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.chat_text_edit)

        self.setStyleSheet("background-color: black;")
        layout.setSizeConstraint(QVBoxLayout.SetDefaultConstraint)
        layout.setStretch(1, 1)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding))

        text_color = QColor(Qt.blue)
        text_color_text = QTextCharFormat()
        text_color_text.setForeground(text_color)
        self.chat_text_edit.setCurrentCharFormat(text_color_text)

        self.gif_label = QLabel()
        self.gif_label.setStyleSheet("border: none;")
        movie = QMovie(GraphicsDirectoryPath('Ai.gif'))
        max_gif_size_W = 480
        max_Gif_size_H = 270
        movie.setScaledSize(QSize(max_gif_size_W, max_Gif_size_H))
        self.gif_label.setAlignment(Qt.AlignRight | Qt.AlignBottom)
        self.gif_label.setMovie(movie)
        movie.start()
        layout.addWidget(self.gif_label)

        self.label = QLabel("")
        self.label.setStyleSheet("color:white; font-size:16px; margin-right: 20px; border:none; margin-top:10px;")
        self.label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.label)

        # Add emotion status label
        self.emotion_label = QLabel("")
        self.emotion_label.setStyleSheet("color:white; font-size:14px; margin-right: 20px;")
        self.emotion_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.emotion_label)

        font = QFont()
        font.setPointSize(13)
        self.chat_text_edit.setFont(font)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.loadMessages)
        self.timer.timeout.connect(self.SpeechRecogText)
        self.timer.start(100)
        self.chat_text_edit.viewport().installEventFilter(self)

        self.setStyleSheet("""
            QScrollBar:vertical {
                border: none;
                background: black;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: white;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                background: black;
                height: 10px;
            }
            QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {
                border: none;
                background: none;
                color: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
        """)

    # Other methods remain unchanged...

        
    def loadMessages(self):
        global old_chat_message
        with open(TempDirectoryPath('Responses.data'),"r", encoding='utf-8') as file:
            messages = file.read()
            
            if None==messages:
                pass
            elif len(messages)<=1:
                pass
            elif str(old_chat_message) == str(messages):
                pass
            else:
                self.addMessage(message=messages, color='White')
                old_chat_message = messages
                    
    def SpeechRecogText(self):
        with open(TempDirectoryPath('Status.data'), "r", encoding='utf-8') as file:
            messages = file.read()
            self.label.setText(messages)
            
    def load_icon(self, path, width=60, height=60):
        pixmap = QPixmap(path)
        new_pixmap = pixmap.scaled(width , height)
        self.icon_label.setPixmap(new_pixmap)
        
    def toggle_icon(self, event=None):
        if self.toggled:
            self.load_icon(GraphicsDirectoryPath('voice.png'),60,60)
            MicButtonInitialed()
            
        else:
            self.load_icon(GraphicsDirectoryPath('mic.png'),60,60)
            MicButtonClosed()
            
        self.toggled = not self.toggled
        
    def addMessage(self, message, color):
        cursor = self.chat_text_edit.textCursor()
        format = QTextCharFormat()
        formatm = QTextBlockFormat()
        formatm.setTopMargin(10)
        formatm.setLeftMargin(10)
        format.setForeground(QColor(color))
        cursor.setCharFormat(format)
        cursor.setBlockFormat(formatm)
        cursor.insertText(message + "\n")
        self.chat_text_edit.setTextCursor(cursor)

    def update_emotion_status(self, emotion: str):
        self.emotion_label.setText(f"Current Emotion: {emotion}")
        
class InitialScreen(QWidget) :
        def __init__(self, parent=None):
            super().__init__(parent)
            desktop = QApplication.desktop()
            screen_width = desktop.screenGeometry().width()
            screen_height = desktop.screenGeometry().height()
            content_layout = QVBoxLayout()
            content_layout.setContentsMargins(0,0,0,0)
            gif_label = QLabel()
            movie = QMovie(GraphicsDirectoryPath('Ai.gif'))
            gif_label.setMovie(movie)
            max_gif_size_H = int(screen_width/ 16*9)
            movie.setScaledSize(QSize(1200, 1100))
            gif_label.setAlignment(Qt.AlignCenter)
            movie.start()
            gif_label.setSizePolicy(QSizePolicy.Expanding , QSizePolicy.Expanding)
            self.icon_label = QLabel()
            pixmap = QPixmap(GraphicsDirectoryPath('sound.png'))
            new_pixmap = pixmap.scaled(60,60)
            self.icon_label.setPixmap(new_pixmap)
            self.icon_label.setFixedSize(150,150)
            self.icon_label.setAlignment(Qt.AlignCenter)
            self.toggled = True
            self.icon_label.mousePressEvent = self.toggle_icon
            self.label = QLabel("")
            self.label.setStyleSheet("color:white; font-size:16px; margin-bottom:0;")
            content_layout.addWidget(gif_label, alignment=Qt.AlignCenter)
            content_layout.addWidget(self.label, alignment=Qt. AlignCenter)
            content_layout.addWidget(self.icon_label, alignment=Qt.AlignCenter)
            content_layout.setContentsMargins(0,0,0,150)
            self.setLayout(content_layout)
            self.setFixedHeight(screen_height)
            self.setFixedWidth(screen_width)
            self.setStyleSheet("background-color: black;")
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.SpeechRecogText)
            self.timer.start(100)
            
        def SpeechRecogText(self):
            with open(TempDirectoryPath('Status.data'), "r", encoding='utf-8') as file:
                messages = file.read()
                self.label.setText(messages)
                
        def load_icon(self,path,width=60,height=60):
            pixmap = QPixmap(path)
            new_pixmap = pixmap.scaled(width , height)
            self.icon_label.setPixmap(new_pixmap)
            
        def toggle_icon(self, event= None):
            if self.toggled:
                self.load_icon(GraphicsDirectoryPath('sound.png'),60,60)
                MicButtonInitialed()
                
            else:
                self.load_icon(GraphicsDirectoryPath('mute.png'), 60,60)
                MicButtonClosed()
                
            self.toggled = not self.toggled
            
class messageScreen(QWidget):
    def __init__(self , parent = None):
        super().__init__(parent)
        desktop = QApplication.desktop()
        screen_width = desktop.screenGeometry().width()
        screen_height = desktop.screenGeometry().height()
        layout = QVBoxLayout()
        label = QLabel("")
        layout.addWidget(label)
        chat_section = ChatSection()
        layout.addWidget(chat_section)
        self.setLayout(layout)
        self.setFixedHeight(screen_height)
        self.setStyleSheet("background-color:black;")
        self.setFixedHeight(screen_height)
        self.setFixedWidth(screen_width)
        
class CustomTopBar(QWidget):
    
    def __init__(self, parent, stacked_widget):
        super().__init__(parent)
        self.offset = None  # Initialize the offset attribute here
        self.InitUI()
        self.current_screen = None
        self.stacked_widget = stacked_widget
        
    def InitUI(self):
        self.setFixedHeight(50)
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignRight)
        home_button = QPushButton()
        home_icon = QIcon(GraphicsDirectoryPath("home-button.png"))
        home_button.setIcon(home_icon)
        home_button.setText(" Home")
        home_button.setStyleSheet("height:40px; line-height:40px; background-color:white; color:black")
        message_button = QPushButton()
        message_icon = QIcon(GraphicsDirectoryPath("chat.png"))
        message_button.setIcon(message_icon)
        message_button.setText(" Chat")
        message_button.setStyleSheet("height:40px; line-height:40px; background-color:white; color:black")
        minimise_button = QPushButton()
        minimise_icon = QIcon(GraphicsDirectoryPath('minimise.png'))
        minimise_button.setIcon(minimise_icon)
        minimise_button.setStyleSheet("background-color:white")
        minimise_button.clicked.connect(self.minimiseWindow)
        self.maximize_button = QPushButton()
        self.maximize_icon = QIcon(GraphicsDirectoryPath('maixmize.png'))
        self.restore_icon = QIcon(GraphicsDirectoryPath('minimize.png'))
        self.maximize_button.setIcon(self.maximize_icon)
        self.maximize_button.setFlat(True)
        self.maximize_button.setStyleSheet("background-color:white")
        self.maximize_button.clicked.connect(self.maximizeWindow)
        close_button = QPushButton()
        close_icon = QIcon(GraphicsDirectoryPath('close.png'))
        close_button.setIcon(close_icon)
        close_button.setStyleSheet("background-color:white")
        close_button.clicked.connect(self.closeWindow)
        line_frame = QFrame()
        line_frame.setFixedHeight(1)
        line_frame.setFrameShape(QFrame.HLine)
        line_frame.setFrameShadow(QFrame.Sunken)
        line_frame.setStyleSheet("border-color: black;")
        title_label = QLabel(f"{str(Assistantname).capitalize()} AI    ")
        title_label.setStyleSheet("color: black;font-size: 18px;; background-color:white")
        home_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        message_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        layout.addWidget(title_label)
        layout.addStretch(1)
        layout.addWidget(home_button)
        layout.addWidget(message_button)
        layout.addStretch(1)
        layout.addWidget(minimise_button)
        layout.addWidget(self.maximize_button)
        layout.addWidget(close_button)
        layout.addWidget(line_frame)
        self.draggable = True
        self.offset = None
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.white)
        super().paintEvent(event)
        
    def minimiseWindow(self):
        self.parent().showMinimized()
        
    def maximizeWindow(self):
        if self.parent().isMaximized():
            self.parent().showNormal()
            self.maximize_button.setIcon(self.maximize_icon)
        else:
            self.parent().showMaximized()
            self.maximize_button.setIcon(self.restore_icon)
            
    def closeWindow(self):
        self.parent().close()
    def mousePressEvent(self, event):
        if self.draggable:
            self.offset = event.pos()
            
    def mouseMoveEvent(self, event):
        if self.draggable and self.offset:
            new_pos = event.globalpos() - self.offset
            
    def showMessageScreen(self):
        if self.current_screen is not None:
            self.current_screen.hide()
            
        message_screen = messageScreen(self)
        layout = self.parent().layout()
        if layout is not None:
            layout.addWidget(message_screen)
            
    def showInitialScreen(self):
        if self.current_screen is not None:
            self.current_screen.hide()
            
        initial_screen = InitialScreen(self)
        layout = self.parent().layout()
        if layout is not None:
            layout.addWidget(initial_screen)
        self.current_screen = initial_screen
        
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.InitUI()
        
    def InitUI(self):
        desktop = QApplication.desktop()
        screen_width = desktop.screenGeometry().width()
        screen_height = desktop.screenGeometry().height()
        stacked_widget = QStackedWidget(self)
        initial_screen = InitialScreen()
        message_screen = messageScreen()
        stacked_widget.addWidget(initial_screen)
        stacked_widget.addWidget(message_screen)
        self.setGeometry(0,0, screen_width , screen_height)
        self.setStyleSheet("background-color:black;")
        top_bar = CustomTopBar(self, stacked_widget)
        self.setMenuWidget(top_bar)
        self.setCentralWidget(stacked_widget)
        
def GraphicalUserInterface():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    GraphicalUserInterface()


