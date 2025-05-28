# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:36:49 2025

@author: Anubhav Prakash
"""

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from itertools import combinations
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation, PillowWriter
import imageio

def get_latest_file(folder_path):
    files=[os.path.join(folder_path,f) for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path,f))]
    if not files:
        raise FileNotFoundError(f"No files found in folder: {folder_path}")
    latest_file =max(files,key=os.path.getmtime)
    return latest_file



def enhance_image(img, contrast_alpha=2.0, brightness_beta=0):
    """
    Given a BGR image:
      1. Convert to HSV and set S channel to 255 (max saturation).
      2. Convert back to BGR.
      3. Apply a linear contrast/brightness adjustment:
         output = img * contrast_alpha + brightness_beta.
    Returns the enhanced BGR image.
    """
    #Max out saturation for ease of reading the image and different color
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s[:] = 255                          # full saturation
    hsv_max = cv2.merge([h, s, v])
    img_sat = cv2.cvtColor(hsv_max, cv2.COLOR_HSV2BGR)

    #increase contrast
    enhanced = cv2.convertScaleAbs(img_sat, alpha=contrast_alpha, beta=brightness_beta)

    return enhanced


def extract_cell_colors(img, grid_size):
    """
    Take a color image and a number grid_size.  Cut the image into grid_size by grid_size smaller squares.
    For each little square, find the average red, green, blue color value by looking at all its pixels.
    Return a list of these average colors for every square, in order from top-left to bottom-right.
    """
    h,w=img.shape[:2]
    cell_h,cell_w= h//grid_size, w//grid_size
    cell_colors =[]

    for i in range(grid_size):
        for j in range(grid_size):
            y1,y2= i*cell_h, (i+1)*cell_h
            x1,x2=j* cell_w, (j+1)*cell_w
            cell_img= img[y1:y2, x1:x2]
            avg_color= np.mean(cell_img.reshape(-1, 3), axis=0)
            cell_colors.append(avg_color)

    return np.array(cell_colors)

def make_init_board(imgpath, grid_size):
    """
    Load an image, compute average colors for a grid of size grid_size, cluster those colors into grid_size groups,
    and build a grid_size×grid_size board where each cell is a dict with:
      - "Color": a label 'C0'..'C{n-1}' for the cluster
      - "state": 'eligible' initially
    Return the board as a list of lists.
    """
    image_bgr=cv2.imread(imgpath)
    if image_bgr is None:
        raise ValueError(f"Could not load image from {imgpath}")
    
    image_bgr = enhance_image(image_bgr, contrast_alpha=1.0)
    image_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    cell_colors=extract_cell_colors(image_rgb, grid_size)

    kmeans=KMeans(n_clusters=grid_size, n_init=10, random_state=42)
    color_labels=kmeans.fit_predict(cell_colors)

    center_order=np.argsort(np.sum(kmeans.cluster_centers_, axis=1))
    label_map={old: new for new,old in enumerate(center_order)}
    remapped_labels=[label_map[label] for label in color_labels]

    board=[]
    idx=0
    for i in range(grid_size):
        row=[]
        for j in range(grid_size):
            row.append({
                "Color": f"C{remapped_labels[idx]}",
                "state": "eligible"
            })
            idx+=1
        board.append(row)

    return board

def draw_bd(board, cell_size=60, frames=None, show_img=True, output_path=None):
    """
    Draw the board as an image. Background color per cell from its 'Color' label,
    'Q' for queens, 'x' for ineligible.  If frames list is provided, append frame;
    otherwise display inline.
    """
    grid_size=len(board)
    img_h=img_w=grid_size*cell_size
    img=np.ones((img_h, img_w, 3), dtype=np.uint8)*255  

    #assigning a unique color to each 
    unique_colors=sorted({cell["Color"] for row in board for cell in row})
    color_map={}
    np.random.seed(42)
    CONTRAST_COLORS = [
        (223, 160, 191),  # pink
        (150, 190, 255),  # light blue
        ( 85, 235, 226),  # cyan
        (230, 243, 136),  # yellow
        (185, 178, 158),  # taupe
        (255, 123,  96),  # coral
        (255, 201, 146),  # peach
        (223, 223, 223),  # light gray
        (149, 203, 207),  # pale teal
        (179, 223, 160),  # light green
        (187, 163, 226),  # lavender
        (249, 241, 221),  # cream
    ]

 
    #assigning fixed colors to labels
    for idx, label in enumerate(unique_colors):
        color_map[label]=CONTRAST_COLORS[idx % len(CONTRAST_COLORS)]
    font=cv2.FONT_HERSHEY_SIMPLEX

    for i in range(grid_size):
        for j in range(grid_size):
            cell=board[i][j]
            color_label=cell["Color"]
            state=cell["state"]

            y1, y2=i*cell_size, (i+1)*cell_size
            x1, x2=j*cell_size, (j+1)*cell_size

            cv2.rectangle(img, (x1, y1), (x2, y2), color_map[color_label], thickness=-1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
            
            if state=="queen":
                cv2.putText(img, "Q", (x1+18, y1+40), font, 1.2, (0, 0, 0), 2)
            elif state=="ineligible":
                cv2.putText(img, "x", (x1+20, y1+40), font, 1.2, (50, 50, 50), 2)

    img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
    

def post_queen_eradication(board, row, col, frames):
    """
    Render the board as an image. Background color per cell from its 'Color' label,
    'Q' for queens, 'x' for ineligible.  If frames list is provided, append frame;
    otherwise display inline.
    """
    n=len(board)
    region_color=board[row][col]["Color"]
    board[row][col]["state"]="queen"
    #row & Column
    for i in range(n):
        if board[row][i]["state"]=="eligible":
            board[row][i]["state"]="ineligible"
            draw_bd(board,  frames=frames)
        if board[i][col]["state"]=="eligible":
            board[i][col]["state"]="ineligible"
            draw_bd(board,  frames=frames)
    # same colored region
    for i in range(n):
        for j in range(n):
            if board[i][j]["Color"]==region_color and board[i][j]["state"]=="eligible":
                board[i][j]["state"]="ineligible"
                draw_bd(board,  frames=frames)
    # neighboring 8 cells
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0:
                continue
            r, c=row+di, col+dj
            if 0 <= r < n and 0 <= c < n and board[r][c]["state"]=="eligible":
                board[r][c]["state"]="ineligible"
                draw_bd(board,  frames=frames)

def build_maps(board):    
    """
    Return two dicts mapping each row and column index to the set of colors of squares
    still marked 'eligible'.
    """
    n=len(board)
    rows={r: set() for r in range(n)}
    cols={c: set() for c in range(n)}
    for i in range(n):
        for j in range(n):
            if board[i][j]["state"]=="eligible":
                rows[i].add(board[i][j]["Color"])
                cols[j].add(board[i][j]["Color"])
    return rows, cols




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

# def eliminate_single_color_entries(board, mapping, is_row):
#     """
#     Given a board and a mapping (rows_map or cols_map) of eligible colors,
#     eliminate cells in other rows/columns when a key has exactly one color.

#     Args:
#         board: List[List[dict]] – n×n board with each cell {"Color": Cx, "state": ...}
#         mapping: Dict[int, Set[str]] – either rows_map or cols_map for eligible cells
#         is_row: bool – True if `mapping` is rows_map, False if cols_map

#     Returns:
#         new_mapping: the updated rows_map or cols_map after elimination
#         changed: bool indicating whether any cell was marked ineligible
#     """
#     n=len(board)
#     changed=True
    
#     while changed:
#         changed=False
#         # For each key (row index or col index) that has exactly one eligible color:
#         for key, colors in list(mapping.items()):
#             if len(colors)==1:
#                 color=next(iter(colors))
#                 # Eliminate that color from all other rows/columns
#                 if is_row:
#                     # for every other row r ≠ key
#                     for r in mapping:
#                         if r==key:
#                             continue
#                         for j in range(n):
#                             cell=board[r][j]
#                             if cell["state"]=="eligible" and cell["Color"]==color:
#                                 cell["state"]="ineligible"
#                                 changed=True
#                                 draw_bd(board,  frames=frames)
#                 else:
#                     # for every other column c ≠ key
#                     for c in mapping:
#                         if c==key:
#                             continue
#                         for i in range(n):
#                             cell=board[i][c]
#                             if cell["state"]=="eligible" and cell["Color"]==color:
#                                 cell["state"]="ineligible"
#                                 changed=True
#                                 draw_bd(board,  frames=frames)

#     # Rebuild both row/col maps and return the one corresponding to is_row
#     new_rows_map, new_cols_map=build_maps(board)
#     return (new_rows_map if is_row else new_cols_map), changed

def candidate_intersection_elimination(board, frames, draw_fn=None, cell_size=60):
    """
    For each color region with at least two 'eligible' squares, collect for each such square
    the set of all 'eligible' squares that would attack it (row, column, neighbors, same color).
    Take the intersection of these sets; if nonempty, mark those squares ineligible and record.
    Return True on first elimination, else False.
    """
    return_val=False
    n=len(board)
    region_map={}
    for i in range(n):
        for j in range(n):
            if board[i][j]["state"]=="eligible":
                region_map.setdefault(board[i][j]["Color"], []).append((i, j))
    
    
    sorted_regions=sorted(region_map.items(), key=lambda kv: len(kv[1]))
    
    for min_color, coords in sorted_regions:
        if len(coords) < 2:
            continue
        
        candidate_sets=[]
        for r, c in coords:
            S=set()
            for x in range(n):
                cell=board[r][x]
                if cell["state"]=="eligible" and cell["Color"] != min_color:
                    S.add((r, x))

            for y in range(n):
                cell=board[y][c]
                if cell["state"]=="eligible" and cell["Color"] != min_color:
                    S.add((y, c))

            for di in (-1,0,1):
                for dj in (-1,0,1):
                    if di==0 and dj==0:
                        continue
                    rr, cc=r+di, c+dj
                    if 0 <= rr < n and 0 <= cc < n:
                        cell=board[rr][cc]
                        if cell["state"]=="eligible" and cell["Color"] != min_color:
                            S.add((rr, cc))
            candidate_sets.append(S)
        

        intersection=set.intersection(*candidate_sets)
        if not intersection:
            continue
        
        for (i, j) in intersection:
            board[i][j]["state"]="ineligible"
            if draw_fn:
                draw_fn(board, cell_size=cell_size, frames=frames)
            
        # print(f"[Candidate elimination] color={min_color}, eliminated={intersection}")
        return_val=True
    
    return return_val


def get_attacking_positions(board, r, c, exclude_row=None, exclude_col=None):
    """
    Return all eligible cells that would attack (r,c) if a queen were placed there,
    excluding any in the given row or column.
    """
    n=len(board)
    positions=set()
    color=board[r][c]["Color"]

    for rr in range(n):
        for cc in range(n):
            # skip the excluded row or column
            if exclude_row is not None and rr==exclude_row:
                continue
            if exclude_col is not None and cc==exclude_col:
                continue

            cell=board[rr][cc]
            if cell["state"] != "eligible":
                continue

            # same column
            if cc==c:
                positions.add((rr, cc))
                continue

            # same row
            if rr==r:
                positions.add((rr, cc))
                continue

            # same region (color)
            if cell["Color"]==color:
                positions.add((rr, cc))
                continue

            # neighbor
            if abs(rr - r) <= 1 and abs(cc - c) <= 1:
                positions.add((rr, cc))
                continue

    return positions


def build_coord_maps(board):
    """
    Build two dicts:
      rows_coords[i]=list of (i,j) for eligible cells in row i
      cols_coords[j]=list of (i,j) for eligible cells in col j
    Skip any row or column that has no eligible cells.
    """
    n=len(board)
    rows_coords={i: [] for i in range(n)}
    cols_coords={j: [] for j in range(n)}
    for i in range(n):
        for j in range(n):
            if board[i][j]["state"]=="eligible":
                rows_coords[i].append((i, j))
                cols_coords[j].append((i, j))
    # filter out empty lists
    rows_coords={i: coords for i, coords in rows_coords.items() if coords}
    cols_coords={j: coords for j, coords in cols_coords.items() if coords}
    return rows_coords, cols_coords

    
def line_candidate_intersection_elimination(board, frames, draw_fn=None):
    """
    Similar to candidate_intersection_elimination but operate on rows then columns.
    For each row/column with at least two 'eligible' squares, find intersection of attack sets
    excluding same row/column itself, then mark those as ineligible if found.
    Return True on first elimination.
    """
    # n=len(board)
    rows_map, cols_map=build_maps(board)
    rows_coords, cols_coords=build_coord_maps(board)

    #process rows by ascending eligible cell count
    for row, coords in sorted(rows_coords.items(), key=lambda kv: len(kv[1])):
        if len(coords) < 2:
            continue
        candidate_sets=[]
        for (r, c) in coords:
            S=get_attacking_positions(board, r, c, exclude_row=row)
            # print(S,r,c, row)
            candidate_sets.append(S)
        intersection=set.intersection(*candidate_sets)
        if intersection:
            for (rr, cc) in intersection:
                board[rr][cc]["state"]="ineligible"
                if draw_fn:
                    draw_fn(board, frames=frames)
            # print(f"[Row‐based elimination] row={row}, eliminated={intersection}")
            return True

    #Process cols
    for col, coords in sorted(cols_coords.items(), key=lambda kv: len(kv[1])):
        if len(coords) < 2:
            continue
        candidate_sets=[]
        for (r, c) in coords:
            S=get_attacking_positions(board, r, c, exclude_col=col)
            candidate_sets.append(S)
        intersection=set.intersection(*candidate_sets)
        if intersection:
            for (rr, cc) in intersection:
                board[rr][cc]["state"]="ineligible"
                if draw_fn:
                    draw_fn(board, frames=frames)
            # print(f"[Column‐based elimination] col={col}, eliminated={intersection}")
            return True

    return False



def eliminate_consecutive_groups(board, rows_map, cols_map, frames, draw_fn=None, cell_size=60, max_k=5):
    """
    For k=2..max_k, check all runs of k consecutive rows then columns:
      - If union of eligible colors== k, eliminate those colors outside block.
      - If union > k and exactly k colors occur only in that block,
        eliminate other colors inside block.
    Return True on first elimination.
    """
    n=len(board)
    max_k=max_k or n

    for mapping, is_row in [(rows_map, True), (cols_map, False)]:
        for k in range(1, min(max_k, n)+1):
            for start in range(n - k+1):
                indices=list(range(start, start+k))
                if any((i not in mapping) or (not mapping[i]) for i in indices):
                    continue

                union_colors=set().union(*(mapping[i] for i in indices))

                #exactly k colors >> prune them outside the lines group
                if len(union_colors)==k:
                    did_elim=False
                    for color in union_colors:
                        for line in range(n):
                            if line in indices:
                                continue
                            for pos in range(n):
                                r, c=(line, pos) if is_row else (pos, line)
                                cell=board[r][c]
                                if cell["state"]=="eligible" and cell["Color"]==color:
                                    cell["state"]="ineligible"
                                    draw_fn(board, cell_size=cell_size, frames=frames)
                                    did_elim=True
                                    return True
        
                    # if did_elim:
                    #     if draw_fn:
                    #         draw_fn(board, cell_size=cell_size, frames=frames)
                    #     kind="rows" if is_row else "cols"
                    #     print(f"[Consec SPECIAL k={k}, idx={indices}, colors={union_colors}, kind={kind}]")
                    #     return True

                #more than k colors >> find confined “special” colors
                if len(union_colors) > k:
                    special={
                        c for c in union_colors
                        #c mustnt appear in any line outside indices
                        if all((line in indices) or (c not in mapping.get(line, ()))
                               for line in range(n))
                    }
                    if len(special)==k:
                        extras=union_colors - special
                        did_elim=False
                        for line in indices:
                            for pos in range(n):
                                r, c=(line, pos) if is_row else (pos, line)
                                cell=board[r][c]
                                if cell["state"]=="eligible" and cell["Color"] in extras:
                                    cell["state"]="ineligible"
                                    did_elim=True
                                    draw_fn(board, cell_size=cell_size, frames=frames)
                                    return True
                        # if did_elim:
                        #     if draw_fn:
                        #         draw_fn(board, cell_size=cell_size, frames=frames)
                        #     kind="rows" if is_row else "cols"
                        #     print(f"[Consec elim k={k}, idx={indices}, union={union_colors}, special={special}, kind={kind}]")
                        #     return True

    return False


def inv_board_chk(board):
    """
    Returns True if the board is in a non-valid state:
      - Any row has no 'eligible' or 'queen' cells (all ineligible).
      - Any column has no 'eligible' or 'queen' cells.
      - Any color region has no 'eligible' or 'queen' cells (all ineligible and no queen placed).
    """
    n=len(board)
    # Check rows
    for i in range(n):
        if all(cell["state"]=="ineligible" for cell in board[i]):
            return True

    #check columns
    for j in range(n):
        if all(board[i][j]["state"]=="ineligible" for i in range(n)):
            return True

    #check color regions
    region_cells={}
    for i in range(n):
        for j in range(n):
            color=board[i][j]["Color"]
            region_cells.setdefault(color,[]).append((i, j))

    for color, coords in region_cells.items():
        #skip regions with a queen
        if any(board[r][c]["state"]=="queen" for r,c in coords):
            continue
        #if all are ineligible (and no queen), invalid
        if all(board[r][c]["state"]=="ineligible" for r,c in coords):
            return True

    return False


def solve_queens(board):
    """
    Solve by iterating: do obvious placements, candidate intersection, line-based intersection,
    and consecutive-group elimination until all queens placed or stuck.  Return final board and frames.
    """
    n=len(board)
    count=0
    frames=[]
    draw_bd(board,  frames=frames)
    queen_count=sum(cell["state"]=="queen" for row in board for cell in row)
    cross_count=sum(cell["state"]=="ineligible" for row in board for cell in row)
    changed=True
    iteration_count=1
    while changed and queen_count < n:
        
        if inv_board_chk(board):
            print("Invalid board : no further moves possible.")
            break       

        print("iter count : ",iteration_count)
        iteration_count +=1
        changed=False

        #simple single-cell placements
        placed, queen_count=do_obv_placing(board, queen_count, draw_fn=draw_bd, frames=frames)
        if placed:
            changed=True
            draw_bd(board,  frames=frames)
            # print("[Changed at check point 0]")
            continue
        if queen_count==n:
            return board


        #
        rows_map, cols_map=build_maps(board)
        # print("Rows Map : \n", rows_map,"\n", "Cols Map: \n", cols_map)
        all_colors={cell["Color"] for row in board for cell in row}

        # #unique color appears in only one row/col
        # rows_map, changed1=eliminate_single_color_entries(board, rows_map, is_row=True)
        # if changed1: 
        #     draw_bd(board,  frames=frames)
        #     # print("changed1")
        # cols_map, changed2=eliminate_single_color_entries(board, cols_map, is_row=False)
        # if changed2: 
        #     draw_bd(board,  frames=frames)
        #     # print("changed2")
        # if changed1 or changed2:
        #     changed=True
        #     draw_bd(board,  frames=frames)
        #     # print("[Conitnue from check point 2]")
        #     continue

        #Combinatorial elimination (pairs, triples, quads)
        for k in range(2, min(5, len(all_colors))):
            # print(f"Checking for {k} combinations")
            for combo in combinations(all_colors, k):
                #rows
                rows_combo=[r for r, vals in rows_map.items() if set(combo).issubset(vals)]
                if len(rows_combo)==k:
                    #check each color in combo appears in exactly those k rows
                    if all(sum(1 for vals in rows_map.values() if color in vals)==k for color in combo):
                        for r in rows_combo:
                            for j in range(n):
                                cell=board[r][j]
                                if cell["state"]=="eligible" and cell["Color"] not in combo:
                                    cell["state"]="ineligible"
                                    changed=True
                                    # print(f"[Rows combo elimination: {combo} in rows {rows_combo}]")
                                    draw_bd(board,  frames=frames)
                                    # print("Changed at checkpoint0.1")
        
                # cols
                cols_combo=[c for c, vals in cols_map.items() if set(combo).issubset(vals)]
                if len(cols_combo)==k:
                    if all(sum(1 for vals in cols_map.values() if color in vals)==k for color in combo):
                        for c in cols_combo:
                            for i in range(n):
                                cell=board[i][c]
                                if cell["state"]=="eligible" and cell["Color"] not in combo:
                                    cell["state"]="ineligible"
                                    changed=True
                                    # print(f"[Cols combo elimination: {combo} in cols {cols_combo}]")
                                    draw_bd(board,  frames=frames)
                                    # print("Changed at checkpoint0.2")
        
            
        if candidate_intersection_elimination(board, draw_fn=draw_bd, frames=frames):
            changed=True
            # print("Changed at checkpoint3")
            continue
        
        if  line_candidate_intersection_elimination(board, draw_fn=draw_bd, frames=frames):
            changed=True
            # print("Changed at checkpoint4")
            continue
        
        if eliminate_consecutive_groups(board, rows_map, cols_map, draw_fn=draw_bd, frames=frames):
            changed=True
            # print("Changed at checkpoint5")
            continue
        if cross_count==sum(cell["state"]=="queen" for row in board for cell in row) and queen_count==sum(cell["state"]=="queen" for row in board for cell in row):
            count +=1
            changed=True
            if count==3:
                # print("cannot solve further")
                # return board
                break

    return board, frames




if __name__=="__main__":
    # imgpath="game.png"  
    screenshot_folder = "C:\\Users\\Anubhav Prakash\\Pictures\\Screenshots"  # Change this to the screenshots folder
    imgpath = get_latest_file(screenshot_folder)
    grid_size=8 
    #In this version, you will need to change this everytime the gridsize of the game changes. Will be sorted out in upcoming versions
    import time
    board=make_init_board(imgpath, grid_size)
    # draw_bd(board, cell_size=60)
    # print(board) 
    # draw_bd(board, cell_size=60, output_path="visualized_board_raw.png")
    
    start = time.perf_counter()
    solved_board , frames= solve_queens(board)
    end = time.perf_counter()
    elapsed = end - start
    # for row in board:
    #    # print([f"{cell['Color']}" for cell in row])
    
    draw_bd(solved_board)
    

    print(f"Total execution time: {elapsed:.3f} seconds")

    writer=imageio.get_writer("queens_solver.mp4",format="FFMPEG", fps=10,codec="libx264",quality=10)
    
    for frame in frames:
        writer.append_data(frame)
    writer.close()


    # fig=plt.figure(figsize=(6,6))
    # im=plt.imshow(frames[0], animated=True)
    # plt.axis('off')
    
    # def update(i):
    #     im.set_array(frames[i])
    #     return (im,)
    
    # ani=FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
    

    # writer=PillowWriter(fps=10) 
    # ani.save("queens_solver.mp4", writer=writer)

