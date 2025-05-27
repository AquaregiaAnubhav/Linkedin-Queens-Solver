# -*- coding: utf-8 -*-
"""
Created on Tue May 27 12:08:28 2025

@author: Anubhav Prakash

Brute‑force solver for LinkedIn Queens game. Uses backtracking with a color‑region heuristic
"""

import cv2
import numpy as np
import copy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import imageio


def extract_cell_colors(img, grid_size):
    h, w = img.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    cell_colors = []

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell_img = img[y1:y2, x1:x2]
            avg_color = np.mean(cell_img.reshape(-1, 3), axis=0)
            cell_colors.append(avg_color)

    return np.array(cell_colors)


def make_init_board(imgpath, grid_size):
    image_bgr = cv2.imread(imgpath)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {imgpath}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cell_colors = extract_cell_colors(image_rgb, grid_size)

    kmeans = KMeans(n_clusters=grid_size, n_init=10, random_state=42)
    labels = kmeans.fit_predict(cell_colors)

    order = np.argsort(np.sum(kmeans.cluster_centers_, axis=1))
    remap = {old: new for new, old in enumerate(order)}
    labels = [remap[l] for l in labels]

    board = []
    idx = 0
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            row.append({
                "Color": f"C{labels[idx]}",
                "state": "eligible"
            })
            idx += 1
        board.append(row)

    return board


def draw_bd(board, cell_size=60, frames=None, show_img=True, output_path=None):
    n = len(board)
    img_h = img_w = n * cell_size
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    unique = sorted({cell["Color"] for row in board for cell in row})
    CONTRAST_COLORS = [
        (223, 160, 191), (150, 190, 255), (85, 235, 226), (230, 243, 136),
        (185, 178, 158), (255, 123,  96), (255, 201, 146), (223, 223, 223),
        (149, 203, 207), (179, 223, 160), (187, 163, 226), (249, 241, 221)
    ]
    cmap = {label: CONTRAST_COLORS[i % len(CONTRAST_COLORS)] for i, label in enumerate(unique)}
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(n):
        for j in range(n):
            cell = board[i][j]
            col = cmap[cell["Color"]]
            y1, y2 = i*cell_size, (i+1)*cell_size
            x1, x2 = j*cell_size, (j+1)*cell_size
            cv2.rectangle(img, (x1,y1), (x2,y2), col, -1)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,0), 1)
            if cell["state"] == "queen":
                cv2.putText(img, "Q", (x1+18, y1+40), font, 1.2, (0,0,0), 2)
            elif cell["state"] == "ineligible":
                cv2.putText(img, "x", (x1+20, y1+40), font, 1.2, (50,50,50), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if frames is not None:
        frames.append(img_rgb)
        return
    if show_img:
        plt.figure()
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.show()
    elif output_path:
        cv2.imwrite(output_path, img)

def do_obv_placing(board, queen_count, frames, draw_fn=None):
    """
    Perform the “obvious” placements:
      – If any row/column/region has exactly one eligible cell, place a queen there.
      – After each placement, optionally call draw_fn(board, frames=frames).
    Returns:
      placed_any (bool): True if at least one queen was placed.
      queen_count (int): updated count of queens on the board.
    """
    n=len(board)
    placed_any=False

    while True:
        placed=False

        # Rows & Columns
        for idx in range(n):
            # Row check
            elig=[(idx, j) for j in range(n) if board[idx][j]["state"]=="eligible"]
            if len(elig)==1:
                r, c=elig[0]
                post_queen_eradication(board, r, c, frames=frames)
                queen_count+=1
                placed=True
                if draw_fn: draw_fn(board, frames=frames)

            # Column check
            elig=[(i, idx) for i in range(n) if board[i][idx]["state"]=="eligible"]
            if len(elig)==1:
                r, c=elig[0]
                post_queen_eradication(board, r, c, frames=frames)
                queen_count+=1
                placed=True
                if draw_fn: draw_fn(board, frames=frames)

        #region (color) check
        region_map={}
        for i in range(n):
            for j in range(n):
                if board[i][j]["state"]=="eligible":
                    region_map.setdefault(board[i][j]["Color"], []).append((i, j))

        for cells in region_map.values():
            if len(cells)==1:
                r, c=cells[0]
                post_queen_eradication(board, r, c, frames=frames)
                queen_count+=1
                placed=True
                if draw_fn: draw_fn(board, frames=frames)

        if not placed:
            break
        placed_any=True

    return placed_any, queen_count

def post_queen_eradication(board, row, col, frames):
    n = len(board)
    region = board[row][col]["Color"]
    board[row][col]["state"] = "queen"

    #eliminate row, column
    for i in range(n):
        if board[row][i]["state"] == "eligible":
            board[row][i]["state"] = "ineligible"
            draw_bd(board, frames=frames)
        if board[i][col]["state"] == "eligible":
            board[i][col]["state"] = "ineligible"
            draw_bd(board, frames=frames)

    # eliminate same-color region
    for i in range(n):
        for j in range(n):
            if board[i][j]["Color"] == region and board[i][j]["state"] == "eligible":
                board[i][j]["state"] = "ineligible"
                draw_bd(board, frames=frames)

    # eliminate neighbors
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            r, c = row+di, col+dj
            if 0<=r<n and 0<=c<n and board[r][c]["state"]=="eligible":
                board[r][c]["state"] = "ineligible"
                draw_bd(board, frames=frames)



def inv_board_chk(board):
    n = len(board)
    #rows
    for i in range(n):
        if all(cell["state"] == "ineligible" for cell in board[i]):
            return True
    #cols
    for j in range(n):
        if all(board[i][j]["state"] == "ineligible" for i in range(n)):
            return True
    #regions
    regions = {}
    for i in range(n):
        for j in range(n):
            regions.setdefault(board[i][j]["Color"], []).append((i,j))
    for color, coords in regions.items():
        if any(board[r][c]["state"]=="queen" for r,c in coords):
            continue
        if all(board[r][c]["state"]=="ineligible" for r,c in coords):
            return True
    return False


def solve_queens(init_board):
    n = len(init_board)
    frames = []
    def backtrack(board, queen_count):
        b = copy.deepcopy(board)
        # b  = board
        
        # mixing obvious placements to pace up the algo
        placed, qcount = do_obv_placing(b, queen_count, frames, draw_fn=draw_bd)
        if inv_board_chk(b):
            return False, None, None
        if qcount == n:
            return True, b, frames

        #choosing color with least eligible cells
        regions = {}
        for i in range(n):
            for j in range(n):
                if b[i][j]["state"] == "eligible":
                    regions.setdefault(b[i][j]["Color"], []).append((i,j))
        color, cells = min(regions.items(), key=lambda kv: len(kv[1]))

        for (r,c) in cells:
            btrial = copy.deepcopy(b)
            # ftrial = copy.deepcopy(frames)
            post_queen_eradication(btrial, r, c, frames)
            if inv_board_chk(btrial):
                continue
            success, sol, sol_frames = backtrack(btrial, qcount+1)
            if success:
                return True, sol, sol_frames
        return False, None, None

    success, board, frames = backtrack(init_board, 0)
    if not success:
        raise ValueError("No solution found")
    return board, frames


if __name__ == "__main__":
    imgpath = "game.png"
    grid_size = 8

    board = make_init_board(imgpath, grid_size)
    
    start = __import__('time').perf_counter()
    
    solved_board, frames = solve_queens(board)
    
    end = __import__('time').perf_counter()
    
    draw_bd(solved_board)
    
    print(f"Total execution time: {end - start:.3f} seconds")

    writer = imageio.get_writer("queens_solver.mp4", format="FFMPEG", fps=10,
                               codec="libx264", quality=10)
    for frm in frames:
        writer.append_data(frm)
    writer.close()
