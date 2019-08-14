import curses
from curses import textpad
import random
import time
from curses import wrapper
import math
import numpy as np


def create_apple(snake, box):
    apple = None

    while apple is None:
        apple = [random.randint(box[0][0] + 1, box[1][0] - 1), random.randint(box[0][1] + 1, box[1][1] - 1)]

        if apple in snake:
            apple = None

    return apple


def print_score(stdscr, score):
    h, w = stdscr.getmaxyx()
    score_text = "Score {}".format(score)
    stdscr.addstr(h - 1, w // 2 - len(score_text) // 2, score_text)
    stdscr.refresh()


def print_angle(stdscr, angle_):
    # angle_ = np.rad2deg(angle_)
    h, w = stdscr.getmaxyx()
    text = "angle {}".format(angle_)
    stdscr.addstr(1, w // 2, text)
    stdscr.refresh()


def print_dir(stdscr, dirList):
    h, w = stdscr.getmaxyx()
    dir = "dir: {0}".format(dirList)
    stdscr.addstr(1, w // 3, dir)
    stdscr.refresh()


def calc_angle(apple_, snake_):
    snake_vect_ = np.array(snake_[0]) - np.array(snake_[1])
    snake_vect = np.copy(snake_vect_)
    snake_vect_[0], snake_vect_[1] = snake_vect_[1], snake_vect_[0]

    apple_vect_ = np.array(apple_) - np.array(snake_[0])
    apple_vect = np.copy(apple_vect_)
    apple_vect_[0], apple_vect_[1] = apple_vect_[1], apple_vect_[0]

    snake_vect_norm_ = np.linalg.norm(snake_vect_)
    apple_vect_norm_ = np.linalg.norm(apple_vect_)

    snake_vect_ = snake_vect_ / snake_vect_norm_
    apple_vect_ = apple_vect_ / apple_vect_norm_

    if apple_vect_norm_ == 0:
        apple_vect_norm_ = 10
    if snake_vect_norm_ == 0:
        snake_vect_norm_ = 10

    # angle = math.acos(np.dot(snake_vect, apple_vect))
    angle = math.atan2(apple_vect_[1] * snake_vect_[0] - apple_vect_[0] * snake_vect_[1],
                       apple_vect_[1] * snake_vect_[1] + apple_vect_[0] * snake_vect_[0]) / math.pi

    return angle, snake_vect, apple_vect


def generate_movement(direction, head_):
    new_head_ = head_
    direction = direction.tolist()
    print("direction: ", direction)
    if direction == [0, 1]:
        # RIGHT
        new_head_ = [head_[0], head_[1] + 1]

    elif direction == [0, -1]:
        # LEFT
        new_head_ = [head_[0], head_[1] - 1]

    elif direction == [-1, 0]:
        # UP
        new_head_ = [head_[0] - 1, head_[1]]

    elif direction == [1, 0]:
        # DOWN
        new_head_ = [head_[0] + 1, head_[1]]

    return new_head_


def generate_dir(stdscr, angle):
    if angle > 0:
        # right
        direction_ = [0, 0, 1]

    elif angle < 0:
        # left
        direction_ = [1, 0, 0]
    else:
        # keep it straight
        direction_ = [0, 1, 0]

    print_dir(stdscr, direction_)

    return direction_

# def is_direction_blocked(snake_position, current_direction_vector, snake_, box):
#     next_step = snake_position[0] + current_direction_vector
#     snake_start = snake_position[0]
#     if  (snake_position[0] in [box[0][0], box[1][0]] or snake_position[1] in [box[0][1], box[1][1]]) == 1 or snake_position in snake_ :
#         return 1
#     else:
#         return 0

def checkBlocked(snake_, snake_dir, new_direction, left_direction_vector, right_direction_vector, box):
    front_blocked = 0
    left_blocked = 0
    right_blocked = 0

    next_pos = snake_[0] + new_direction
    print(type(snake_[0]))
    print(type(next_pos.tolist()))
    # next_pos.tolist()
    next_pos = next_pos.tolist()
    print("act pos: ", snake_[0])
    print("next pos: ", next_pos)
    print("box: ", box[0][0], box[1][0], box[0][1], box[1][1])

    if snake_dir.tolist() == new_direction.tolist():
        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            front_blocked = 1
        else:
            front_blocked = 0

    if new_direction.tolist() == left_direction_vector.tolist():
        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            left_blocked = 1
        else:
            left_blocked = 0

    if new_direction.tolist() == right_direction_vector.tolist():
        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            right_blocked = 1
        else:
            right_blocked = 0

    # front_blocked = is_direction_blocked(next_pos, snake_dir, snake_, box)
    # left_blocked = is_direction_blocked(next_pos, left_direction_vector, snake_, box)
    # right_blocked = is_direction_blocked(next_pos, right_direction_vector, snake_, box)

    return  left_blocked, front_blocked, right_blocked


def main(stdscr):
    curses.curs_set(0)  # stop cursor from blink
    stdscr.nodelay(1)  # getch method will not block the code anymore
    stdscr.timeout(150)  # waits 150 ms till next loop
    run = 0
    runNumber = 1

    input_vect = []
    output_vect = []

    while run < runNumber:

        h, w = stdscr.getmaxyx()  # get window size
        box = [[3, 3], [h - 3, w - 3]]  # define game space with textpad
        textpad.rectangle(stdscr, box[0][0], box[0][1], box[1][0], box[1][1])

        snake = [[h // 2, w // 2 + 1], [h // 2, w // 2], [h // 2, w // 2 - 1]]

        for y, x in snake:
            stdscr.addstr(y, x, "#")

        apple = create_apple(snake, box)
        stdscr.addstr(apple[0], apple[1], '*')

        angle, snake_dir, apple_dir = calc_angle(apple, snake)
        print_angle(stdscr, angle)

        score = 0
        print_score(stdscr, score)

        while 1:

            head = snake[0]

            angle, snake_dir, apple_dir = calc_angle(apple, snake)
            direction = generate_dir(stdscr, angle)

            left_direction_vector = np.array([-snake_dir[1], snake_dir[0]])
            right_direction_vector = np.array([snake_dir[1], -snake_dir[0]])

            new_direction = snake_dir
            if direction == [1, 0, 0]:
                new_direction = left_direction_vector
            if direction == [0, 0, 1]:
                new_direction = right_direction_vector

            new_head = generate_movement(new_direction, head)
            # print(new_direction, left_direction_vector, right_direction_vector)

            left_blocked, front_blocked, right_blocked  = checkBlocked(snake, snake_dir, new_direction, left_direction_vector,
                                        right_direction_vector, box)

            print("block vect: ", left_blocked, front_blocked, right_blocked)

            if left_blocked and right_blocked and front_blocked:
                break

            input_vect.append([front_blocked, left_blocked, right_blocked, apple_dir[0], apple_dir[1],
                              snake_dir[0], snake_dir[1]])

            output_vect.append(direction)

            print_angle(stdscr, angle)

            angle, snake_dir, apple_dir = calc_angle(apple, snake)

            snake.insert(0, new_head)
            stdscr.addstr(new_head[0], new_head[1], "#")

            if snake[0] == apple:
                score += 1
                print_score(stdscr, score)

                apple = create_apple(snake, box)
                stdscr.addstr(apple[0], apple[1], '*')

            else:
                stdscr.addstr(snake[-1][0], snake[-1][1], " ")
                snake.pop()

            if (snake[0][0] in [box[0][0], box[1][0]] or
                    snake[0][1] in [box[0][1], box[1][1]] or
                    snake[0] in snake[1:]):
                # msg = "Game over!"

                # stdscr.addstr(h // 2, w // 2 - len(msg), msg)
                stdscr.nodelay(0)
                stdscr.getch()
                time.sleep(3)
                stdscr.clear()
                stdscr.refresh()
                break
            time.sleep(0.01)
            stdscr.refresh()
        run += 1
    stdscr.getch()


wrapper(main)
