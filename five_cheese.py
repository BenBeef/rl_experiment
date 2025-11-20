# coding:utf-8
import pygame
import sys

# 初始化参数
BOARD_SIZE = 15  # 15x15 棋盘
CELL_SIZE = 40  # 每个格子像素大小
CHESS_RADIUS = 18  # 棋子半径
BG_COLOR = (220, 179, 92)  # 棋盘背景色

# 初始化 pygame
pygame.init()
screen = pygame.display.set_mode((BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
pygame.display.set_caption("五子棋")

# 初始化棋盘（0=空，1=黑，2=白）
board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
current_player = 1  # 当前玩家（黑方先行）


def draw_board():
    """绘制棋盘"""
    screen.fill(BG_COLOR)
    # 绘制网格线
    for i in range(BOARD_SIZE):
        pygame.draw.line(screen, (0, 0, 0),
                         (i * CELL_SIZE + CELL_SIZE // 2, CELL_SIZE // 2),
                         (i * CELL_SIZE + CELL_SIZE // 2, (BOARD_SIZE - 1) * CELL_SIZE + CELL_SIZE // 2))
        pygame.draw.line(screen, (0, 0, 0),
                         (CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2),
                         ((BOARD_SIZE - 1) * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
    # 绘制棋子
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 1:
                pygame.draw.circle(screen, (0, 0, 0),
                                   (i * CELL_SIZE + CELL_SIZE // 2, j * CELL_SIZE + CELL_SIZE // 2), CHESS_RADIUS)
            elif board[i][j] == 2:
                pygame.draw.circle(screen, (255, 255, 255),
                                   (i * CELL_SIZE + CELL_SIZE // 2, j * CELL_SIZE + CELL_SIZE // 2), CHESS_RADIUS)


def check_win(x, y):
    """检查是否获胜"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 四个检查方向
    for dx, dy in directions:
        count = 1
        # 正向检查
        i, j = x + dx, y + dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and board[i][j] == current_player:
            count += 1
            i += dx
            j += dy
        # 反向检查
        i, j = x - dx, y - dy
        while 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and board[i][j] == current_player:
            count += 1
            i -= dx
            j -= dy
        if count >= 5:
            return True
    return False


# 游戏主循环
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # 获取鼠标点击位置
            x, y = event.pos
            # 转换为棋盘坐标
            i = round((x - CELL_SIZE // 2) / CELL_SIZE)
            j = round((y - CELL_SIZE // 2) / CELL_SIZE)
            if 0 <= i < BOARD_SIZE and 0 <= j < BOARD_SIZE and board[i][j] == 0:
                board[i][j] = current_player
                if check_win(i, j):
                    print(f"玩家 {current_player} 获胜！")
                    running = False
                current_player = 2 if current_player == 1 else 1

    draw_board()
    pygame.display.flip()

pygame.quit()
sys.exit()
