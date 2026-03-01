import numpy as np

class GomokuBoard:
    """
    Simplified Gomoku (Five in a Row) on a 10x10 board.
    Players: 1 (AI) and 2 (opponent). 0 means empty.
    """
    def __init__(self, size=10):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.num_actions = size * size
        self.step_penalty = -0.01          # 每步惩罚
        self.block_reward = 0.5             # 堵截奖励

    def reset(self):
        self.board.fill(0)
        return self.get_state()

    def get_state(self):
        return tuple(map(tuple, self.board))

    def get_valid_actions(self):
        return [i for i in range(self.num_actions) if self.board[i // self.size][i % self.size] == 0]

    def place_piece(self, action, player):
        row = action // self.size
        col = action % self.size
        self.board[row][col] = player

    def check_winner(self, action, player):
        if action is None:
            return False
        size = self.size
        row = action // size
        col = action % size
        board = self.board
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            r, c = row + dr, col + dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r += dr
                c += dc
            r, c = row - dr, col - dc
            while 0 <= r < size and 0 <= c < size and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 5:
                return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def is_blocking_move(self, action, player):
        """
        检查当前落子是否堵住了对手的潜在胜利（对手已有三子或四子连珠）。
        """
        opponent = 3 - player
        size = self.size
        r, c = divmod(action, size)
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dr, dc in directions:
            cnt = 0
            # 正方向
            rr, cc = r + dr, c + dc
            while 0 <= rr < size and 0 <= cc < size and self.board[rr][cc] == opponent:
                cnt += 1
                rr += dr
                cc += dc
            # 负方向
            rr, cc = r - dr, c - dc
            while 0 <= rr < size and 0 <= cc < size and self.board[rr][cc] == opponent:
                cnt += 1
                rr -= dr
                cc -= dc
            if cnt >= 3:      # 对手已有三子或更多，说明这一步是堵截
                return True
        return False

    def step(self, action, player):
        """
        Execute action for player.
        Returns: next_state, reward (for this player), done.
        reward: +1 if player wins, +0.5 for blocking opponent's threat,
                -0.01 for each non-terminal move, 0 if draw.
        """
        self.place_piece(action, player)
        done = False
        reward = 0

        if self.check_winner(action, player):
            reward = 1
            done = True
        elif self.is_draw():
            done = True
            reward = 0
        else:
            reward = self.step_penalty
            if self.is_blocking_move(action, player):
                reward += self.block_reward

        next_state = self.get_state()
        return next_state, reward, done