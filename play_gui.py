import tkinter as tk
from tkinter import messagebox
import numpy as np
import pickle
import os
from gomoku_game import GomokuBoard
from q_learning_agent import QLearningAgent

class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku Challenger")
        
        self.board_size = 10
        self.cell_size = 60
        self.canvas_size = self.board_size * self.cell_size
        self.offset = 20
        
        self.env = GomokuBoard(size=self.board_size)
        self.agent = QLearningAgent(board_size=self.board_size, epsilon=0.05)
        
        self.state = None
        self.done = False
        self.current_player = 1
        self.last_reward = 0.0
        self.last_action = None
        self.highlight_item = None
        self.game_state = 'start'  # 'start', 'playing', 'gameover'
        self.winner_text = ""
        
        # 画布
        self.canvas = tk.Canvas(root, width=self.canvas_size + 2*self.offset, 
                                height=self.canvas_size + 2*self.offset, bg='burlywood')
        self.canvas.pack(pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # 信息显示区域（仅用于 playing 状态）
        self.info_frame = tk.Frame(root)
        self.info_frame.pack(pady=5)
        
        self.turn_label = tk.Label(self.info_frame, text="Turn: AI (Black)", font=('Arial', 14))
        self.turn_label.pack(side=tk.LEFT, padx=20)
        
        self.reward_label = tk.Label(self.info_frame, text="Last Reward: 0.00", font=('Arial', 14))
        self.reward_label.pack(side=tk.LEFT, padx=20)
        
        self.status_label = tk.Label(root, text="", font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        self.restart_btn = tk.Button(root, text="New Game", command=self.restart_game, font=('Arial', 12))
        self.restart_btn.pack(pady=5)
        
        # 加载Q表
        if os.path.exists("q_table.pkl"):
            with open("q_table.pkl", "rb") as f:
                self.agent.q_table = pickle.load(f)
            self.status_label.config(text=f"Loaded Q-table with {len(self.agent.q_table)} states")
        else:
            self.status_label.config(text="No Q-table found. AI will act randomly.")
        
        # 绑定键盘事件
        self.root.bind('<Key>', self.on_key_press)
        
        # 绘制开始界面
        self.draw_board()
    
    def draw_board(self):
        """根据当前游戏状态绘制棋盘或界面"""
        self.canvas.delete("all")
        
        if self.game_state == 'start':
            # 开始界面：不画棋盘，只显示欢迎信息
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 - 60,
                                    text="Gomoku Challenger", font=('Arial', 36, 'bold'), fill='navy')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 - 10,
                                    text="Click on a cell to place your black stone",
                                    font=('Arial', 16), fill='black')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 + 20,
                                    text="AI plays white stones",
                                    font=('Arial', 16), fill='black')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 + 50,
                                    text="First to five in a row wins",
                                    font=('Arial', 16), fill='black')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 + 100,
                                    text="Press G to start, ESC to quit!",
                                    font=('Arial', 16, 'bold'), fill='red')
        elif self.game_state == 'playing':
            # 正常游戏：绘制棋盘和棋子
            for i in range(self.board_size):
                x0 = self.offset + i * self.cell_size
                y0 = self.offset
                self.canvas.create_line(x0, y0, x0, self.canvas_size + self.offset, fill='black')
                self.canvas.create_line(self.offset, self.offset + i * self.cell_size,
                                        self.canvas_size + self.offset, self.offset + i * self.cell_size, fill='black')
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.env.board[r][c] != 0:
                        x = self.offset + c * self.cell_size
                        y = self.offset + r * self.cell_size
                        color = 'black' if self.env.board[r][c] == 1 else 'white'
                        self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=color, outline='gray', width=2)
            
            # 显示当前回合和奖励
            self.turn_label.config(text=f"Turn: {'AI (Black)' if self.current_player == 1 else 'You (White)'}")
            self.reward_label.config(text=f"Last Reward: {self.last_reward:.2f}")
        
        elif self.game_state == 'gameover':
            # 游戏结束界面：先绘制棋盘，再覆盖半透明层和文字
            # 绘制棋盘
            for i in range(self.board_size):
                x0 = self.offset + i * self.cell_size
                y0 = self.offset
                self.canvas.create_line(x0, y0, x0, self.canvas_size + self.offset, fill='black')
                self.canvas.create_line(self.offset, self.offset + i * self.cell_size,
                                        self.canvas_size + self.offset, self.offset + i * self.cell_size, fill='black')
            
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if self.env.board[r][c] != 0:
                        x = self.offset + c * self.cell_size
                        y = self.offset + r * self.cell_size
                        color = 'black' if self.env.board[r][c] == 1 else 'white'
                        self.canvas.create_oval(x-12, y-12, x+12, y+12, fill=color, outline='gray', width=2)
            
            # 半透明遮罩
            self.canvas.create_rectangle(self.offset, self.offset,
                                         self.canvas_size + self.offset, self.canvas_size + self.offset,
                                         fill='gray80', stipple='gray50', outline='')
            
            # 游戏结束文字
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 - 40,
                                    text="Game Over", font=('Arial', 36, 'bold'), fill='red')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 + 10,
                                    text=self.winner_text, font=('Arial', 24), fill='darkred')
            self.canvas.create_text(self.canvas_size//2 + self.offset, self.canvas_size//2 + 60,
                                    text="Press R to restart, ESC to quit",
                                    font=('Arial', 16), fill='black')
    
    def on_key_press(self, event):
        """处理键盘按键"""
        key = event.keysym
        if key == 'Escape':
            self.root.quit()
        elif key == 'g' or key == 'G':
            if self.game_state == 'start':
                self.start_game()
        elif key == 'r' or key == 'R':
            self.restart_game()
    
    def start_game(self):
        """开始新游戏（从开始界面进入）"""
        self.state = self.env.reset()
        self.done = False
        self.current_player = 1
        self.last_reward = 0.0
        self.last_action = None
        self.game_state = 'playing'
        self.draw_board()
        if self.current_player == 1:
            self.root.after(500, self.ai_move)
    
    def restart_game(self):
        """重新开始游戏（可用于任何状态）"""
        self.start_game()
    
    def on_mouse_move(self, event):
        if self.game_state != 'playing' or self.done or self.current_player != 2:
            if self.highlight_item:
                self.canvas.delete(self.highlight_item)
                self.highlight_item = None
            return
        
        x, y = event.x, event.y
        if (self.offset <= x <= self.canvas_size + self.offset and 
            self.offset <= y <= self.canvas_size + self.offset):
            col = (x - self.offset) // self.cell_size
            row = (y - self.offset) // self.cell_size
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                cx = self.offset + col * self.cell_size
                cy = self.offset + row * self.cell_size
                if self.highlight_item:
                    self.canvas.delete(self.highlight_item)
                self.highlight_item = self.canvas.create_rectangle(
                    cx - self.cell_size//2, cy - self.cell_size//2,
                    cx + self.cell_size//2, cy + self.cell_size//2,
                    outline='red', width=3, dash=(4, 4)
                )
                return
        if self.highlight_item:
            self.canvas.delete(self.highlight_item)
            self.highlight_item = None
    
    def on_canvas_click(self, event):
        if self.game_state != 'playing' or self.done or self.current_player != 2:
            return
        
        x, y = event.x, event.y
        if (self.offset <= x <= self.canvas_size + self.offset and 
            self.offset <= y <= self.canvas_size + self.offset):
            col = (x - self.offset) // self.cell_size
            row = (y - self.offset) // self.cell_size
            if 0 <= row < self.board_size and 0 <= col < self.board_size:
                action = row * self.board_size + col
                valid_actions = self.env.get_valid_actions()
                if action in valid_actions:
                    next_state, reward, done = self.env.step(action, player=2)
                    self.last_reward = reward
                    self.last_action = action
                    self.state = next_state
                    self.done = done
                    
                    self.draw_board()
                    self.reward_label.config(text=f"Last Reward: {reward:.2f}")
                    
                    if self.highlight_item:
                        self.canvas.delete(self.highlight_item)
                        self.highlight_item = None
                    
                    if done:
                        self.show_game_over()
                    else:
                        self.current_player = 1
                        self.turn_label.config(text="Turn: AI (Black)")
                        self.root.after(500, self.ai_move)
                else:
                    self.status_label.config(text="Invalid move! That cell is occupied.")
    
    def ai_move(self):
        if self.game_state != 'playing' or self.done or self.current_player != 1:
            return
        
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            self.done = True
            self.show_game_over()
            return
        
        action = self.agent.get_action(self.state, valid_actions)
        next_state, reward, done = self.env.step(action, player=1)
        self.last_reward = reward
        self.last_action = action
        self.state = next_state
        self.done = done
        
        self.draw_board()
        self.reward_label.config(text=f"Last Reward: {reward:.2f}")
        self.status_label.config(text=f"AI chose ({action//self.board_size}, {action%self.board_size})")
        
        if done:
            self.show_game_over()
        else:
            self.current_player = 2
            self.turn_label.config(text="Turn: You (White)")
    
    def show_game_over(self):
        """游戏结束时调用，显示结束界面"""
        if self.env.check_winner(self.last_action, 1):
            self.winner_text = "AI Wins!"
        elif self.env.check_winner(self.last_action, 2):
            self.winner_text = "You Win!"
        else:
            self.winner_text = "It's a Draw!"
        
        self.game_state = 'gameover'
        self.draw_board()
        # 不再弹出 messagebox，直接显示界面

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()