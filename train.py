import random
import numpy as np
import pickle
import os
import time
import tkinter as tk
from tkinter import messagebox
from gomoku_game import GomokuBoard
from q_learning_agent import QLearningAgent

def get_winner(board):
    """检查五子连珠，返回胜者（1或2）或0（无）"""
    size = board.shape[0]
    # 行
    for r in range(size):
        for c in range(size - 4):
            if board[r, c] != 0 and all(board[r, c + off] == board[r, c] for off in range(5)):
                return board[r, c]
    # 列
    for c in range(size):
        for r in range(size - 4):
            if board[r, c] != 0 and all(board[r + off, c] == board[r, c] for off in range(5)):
                return board[r, c]
    # 正斜
    for r in range(size - 4):
        for c in range(size - 4):
            if board[r, c] != 0 and all(board[r + off, c + off] == board[r, c] for off in range(5)):
                return board[r, c]
    # 反斜
    for r in range(size - 4):
        for c in range(4, size):
            if board[r, c] != 0 and all(board[r + off, c - off] == board[r, c] for off in range(5)):
                return board[r, c]
    return 0

class TrainingGUI:
    """训练可视化窗口"""
    def __init__(self, root, board_size=10):
        self.root = root
        self.root.title("AI Training - Self Play (10x10)")
        
        self.board_size = board_size
        self.cell_size = 50
        self.canvas_size = self.board_size * self.cell_size
        self.offset = 20
        
        # 画布
        self.canvas = tk.Canvas(root, width=self.canvas_size + 2*self.offset, 
                                height=self.canvas_size + 2*self.offset, bg='burlywood')
        self.canvas.pack(pady=10)
        
        # 信息显示
        info_frame = tk.Frame(root)
        info_frame.pack(pady=5)
        
        self.episode_label = tk.Label(info_frame, text="Episode: 0", font=('Arial', 12))
        self.episode_label.pack(side=tk.LEFT, padx=10)
        
        self.player_label = tk.Label(info_frame, text="Current: Black", font=('Arial', 12))
        self.player_label.pack(side=tk.LEFT, padx=10)
        
        self.reward_label = tk.Label(info_frame, text="Last Reward: 0.00", font=('Arial', 12))
        self.reward_label.pack(side=tk.LEFT, padx=10)
        
        self.status_label = tk.Label(root, text="Training in progress...", font=('Arial', 12))
        self.status_label.pack(pady=5)
        
        self.root.update()
    
    def draw_board(self, board, last_action=None, reward=None, episode=None, current_player=None):
        """绘制当前棋盘"""
        self.canvas.delete("all")
        # 网格线
        for i in range(self.board_size):
            x0 = self.offset + i * self.cell_size
            y0 = self.offset
            self.canvas.create_line(x0, y0, x0, self.canvas_size + self.offset, fill='black')
            self.canvas.create_line(self.offset, self.offset + i * self.cell_size,
                                    self.canvas_size + self.offset, self.offset + i * self.cell_size, fill='black')
        
        # 棋子
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r][c] != 0:
                    x = self.offset + c * self.cell_size
                    y = self.offset + r * self.cell_size
                    color = 'black' if board[r][c] == 1 else 'white'
                    self.canvas.create_oval(x-10, y-10, x+10, y+10, fill=color, outline='gray', width=2)
        
        # 更新标签
        if episode is not None:
            self.episode_label.config(text=f"Episode: {episode}")
        if current_player is not None:
            player_str = "Black (AI1)" if current_player == 1 else "White (AI2)"
            self.player_label.config(text=f"Current: {player_str}")
        if reward is not None:
            self.reward_label.config(text=f"Last Reward: {reward:.2f}")
        
        self.root.update()
    
    def set_status(self, text):
        self.status_label.config(text=text)
        self.root.update()

def train(episodes=100, board_size=10, render_every=1, save_every=20, 
          save_path="q_table.pkl", gui=True, step_delay=0.01):
    """
    gui: 是否显示训练可视化窗口
    step_delay: 每一步后的延迟（秒），便于观察
    """
    env = GomokuBoard(size=board_size)
    agent = QLearningAgent(board_size=board_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.3)

    # 加载已有Q表
    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Loaded existing Q-table from {save_path} (contains {len(agent.q_table)} states)")

    win_counts = []
    loss_counts = []
    draw_counts = []

    # 初始化GUI
    if gui:
        root = tk.Tk()
        gui_window = TrainingGUI(root, board_size=board_size)
        root.update()
    else:
        gui_window = None

    for ep in range(episodes):
        state = env.reset()
        done = False
        current_player = 1  # 黑棋先手
        agent.epsilon = max(0.01, agent.epsilon * 0.999)

        # 一局自我对弈
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            action = agent.get_action(state, valid_actions)
            next_state, reward, done = env.step(action, player=current_player)
            agent.learn(state, action, reward, next_state, done)

            # 更新GUI（如果启用）
            if gui_window:
                gui_window.draw_board(env.board, last_action=action, reward=reward,
                                      episode=ep+1, current_player=current_player)
                if step_delay > 0:
                    time.sleep(step_delay)
                # 检查窗口是否被关闭
                try:
                    gui_window.root.update()
                except tk.TclError:
                    print("GUI window closed. Exiting training.")
                    return agent, win_counts, loss_counts, draw_counts

            state = next_state
            if not done:
                current_player = 3 - current_player

        # 统计结果
        winner = get_winner(env.board)
        if winner == 1:
            win_counts.append(1)
            loss_counts.append(0)
            draw_counts.append(0)
        elif winner == 2:
            win_counts.append(0)
            loss_counts.append(1)
            draw_counts.append(0)
        else:
            win_counts.append(0)
            loss_counts.append(0)
            draw_counts.append(1)

        # 定期保存
        if (ep + 1) % save_every == 0:
            with open(save_path, "wb") as f:
                pickle.dump(agent.q_table, f)
            msg = f"Episode {ep+1}: Q-table saved ({len(agent.q_table)} states)"
            print(msg)
            if gui_window:
                gui_window.set_status(msg)

        # 输出进度
        if (ep + 1) % render_every == 0:
            recent = 100
            win_rate = np.sum(win_counts[-recent:]) / min(recent, len(win_counts))
            loss_rate = np.sum(loss_counts[-recent:]) / min(recent, len(loss_counts))
            draw_rate = np.sum(draw_counts[-recent:]) / min(recent, len(draw_counts))
            msg = f"Episode {ep+1}: Win rate {win_rate:.2f}, Loss {loss_rate:.2f}, Draw {draw_rate:.2f}, Epsilon {agent.epsilon:.3f}"
            print(msg)
            if gui_window:
                gui_window.set_status(msg)

    # 最终保存
    with open(save_path, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Training completed. Final Q-table saved to {save_path} (contains {len(agent.q_table)} states)")
    if gui_window:
        gui_window.set_status("Training completed!")
        # 保持窗口打开，等待用户关闭
        gui_window.root.mainloop()

    return agent, win_counts, loss_counts, draw_counts

if __name__ == "__main__":
    # 默认启用GUI，步延迟0.1秒以便观察
    trained_agent, wins, losses, draws = train(episodes=100, save_every=20, gui=True, step_delay=0.01)