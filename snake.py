import curses
from curses import textpad
import random
import time
import math
import numpy as np
import logging


class Snake(object):

    def __init__(self, gui=False):
        self.gui = gui
        self.speed = 0.1

        self.h = 20
        self.w = 20
        self.box = [[3, 3], [self.h - 3, self.w - 3]]  # define game space with textpad

    def snake_init(self, h, w):
        snake_ = [[h // 2, w // 2 + 1], [h // 2, w // 2], [h // 2, w // 2 - 1]]
        return snake_

    def play(self, testNN=False, _model=None):
        """
        if testNN = True, you gonna bypass the decision making algorithm and feed the snake with predictions
        """
#         logger = logging.getLogger(__file__)
#         hdlr = logging.FileHandler(__file__ + ".log")
#         formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
#         hdlr.setFormatter(formatter)
#         logger.addHandler(hdlr)
#         logger.setLevel(logging.DEBUG)

        box = self.box
        apple = None
        snake = self.snake_init(self.h, self.w)
        apple = self.create_apple(apple, snake, box)

        input_vect = []
        output_vect = []
        direction = [0, 1, 0]
        score = 0
        prev_score = 0
        dist = self.getAppleDistance(apple, snake)
        prev_dist = dist
        
        while 1:
            if self.gui:
                stdscr = self.init_render()
                self.render(stdscr, self.box, snake, apple)
                self.print_score(stdscr, score)
                self.print_dir(stdscr, direction)
                time.sleep(self.speed)

            head = snake[0]
            angle, snake_dir, apple_dir = self.calc_angle(apple, snake)
            direction = self.generate_dir(angle)
            left_direction_vector = np.array([snake_dir[1]*(-1), snake_dir[0]])
            right_direction_vector = np.array([snake_dir[1], snake_dir[0]*(-1)])
            left_blocked, front_blocked, right_blocked = self.checkBlocked(snake, snake_dir, left_direction_vector, right_direction_vector, box)
            new_direction = snake_dir.copy()
#             print(angle)
            # --------------------------------------------------------------------------------------------
            if testNN == False:
             
                direction = self.dir_correction(direction, new_direction, left_blocked, front_blocked, right_blocked)  
                dirScalar = 0          
                if direction == [1, 0, 0]:
                    new_direction = left_direction_vector
                    dirScalar = -1
                elif direction == [0, 0, 1]:
                    new_direction = right_direction_vector
                    dirScalar = 1
             
                if left_blocked and front_blocked and right_blocked:
                 break
                
                input_test_vector = [left_blocked, front_blocked, right_blocked, angle, dirScalar]
                input_vect.append(input_test_vector)
                
                if score > prev_score or dist < prev_dist:
                    output = 1
                else:
                    output = 0
                    

            else:

                if direction == [1, 0, 0]:
                    new_direction = left_direction_vector
                if direction == [0, 0, 1]:
                    new_direction = right_direction_vector
                
                predicted_directions = []
                for dirScalar in range(-1, 2):
                    input_test_vector = [left_blocked, front_blocked, right_blocked, angle, dirScalar]

#                 input_test_vector = np.array(input_test_vector).reshape(-1, 7)  # creating a vector from a list KERAS
                    input_test_vector = np.array(input_test_vector).reshape(-1, 5, 1)  # creating a vector from a list tflearn
                    predicted_direction = self.NN(_model, input_test_vector)
                    predicted_directions.append(predicted_direction)

                input_vect.append(input_test_vector)
                predicted_direction_index = np.argmax(np.array(predicted_directions))-1

                if predicted_direction_index == -1:
                    new_direction = left_direction_vector
                if predicted_direction_index == 1:
                    new_direction = right_direction_vector

            # --------------------------------------------------------------------------------------------
            new_head = self.generate_movement(new_direction, head)

            snake.insert(0, new_head)

            if snake[0] == apple:
                score += 1
                apple = None
                apple = self.create_apple(apple, snake, box)

            else:
                snake.pop()

            # --------------------------------------------------------------------------------------------
            if (snake[0][0] in [box[0][0], box[1][0]] or
                    snake[0][1] in [box[0][1], box[1][1]] or
                    snake[0] in snake[1:]):
                # msg = "Game over!"
                output = -1
                output_vect.append(output)
                time.sleep(0.5)
                if self.gui:
                    self.quit_render(stdscr)
                break


            prev_score = score
            prev_dist = dist
            if testNN == False:
                output_vect.append(output)
            
        return input_vect, output_vect, score

    def print_score(self, stdscr, score):
        #         h, w = stdscr.getmaxyx()
        score_text = "Score {}".format(score)
        stdscr.addstr(self.h - 2, self.w // 2 - len(score_text) // 2, score_text)
        stdscr.refresh()

    def print_dir(self, stdscr, dirList):
        dir_ = "dir: {0}".format(dirList)
        stdscr.addstr(2, self.w // 2 - len(dir_) // 2, dir_)
        stdscr.refresh()

    def create_apple(self, apple, snake, box):
        """
        checks if apple in snake -- if true, create apple inside box
        """
        while apple is None:
            apple = [random.randint(box[0][0] + 1, box[1][0] - 1), random.randint(box[0][1] + 1, box[1][1] - 1)]

            if apple in snake:
                apple = None

        return apple

    def getAppleDistance(self, apple_, snake_):
        apple_vect_ = np.array(apple_) - np.array(snake_[0])
        distance = np.linalg.norm(apple_vect_)
        return distance

    def calc_angle(self, apple_, snake_):
        snake_vect_ = np.array(snake_[0]) - np.array(snake_[1])
        snake_vect_[0], snake_vect_[1] = snake_vect_[1], snake_vect_[0]

        apple_vect_ = np.array(apple_) - np.array(snake_[0])
        apple_vect_[0], apple_vect_[1] = apple_vect_[1], apple_vect_[0]

        snake_vect_norm_ = np.linalg.norm(snake_vect_)
        apple_vect_norm_ = np.linalg.norm(apple_vect_)

        snake_vect_ = snake_vect_ / snake_vect_norm_
        apple_vect_ = apple_vect_ / apple_vect_norm_

        angle = math.atan2(apple_vect_[1] * snake_vect_[0] - apple_vect_[0] * snake_vect_[1],
                           apple_vect_[1] * snake_vect_[1] + apple_vect_[0] * snake_vect_[0]) / math.pi

        apple_vect_[1], apple_vect_[0] = apple_vect_[0], apple_vect_[1]
        snake_vect_[1], snake_vect_[0] = snake_vect_[0], snake_vect_[1]
        
        return angle, snake_vect_, apple_vect_

    def generate_movement(self, direction, head_):
        new_head_ = head_
        direction = direction.tolist()
        #     print("direction: ", direction)
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

    def generate_dir(self, angle):
        if angle > 0:
            # right
            direction_ = [0, 0, 1]

        elif angle < 0:
            # left
            direction_ = [1, 0, 0]
        else:
            # keep it straight
            direction_ = [0, 1, 0]

        #         print_dir(stdscr, direction_)

        return direction_

    def checkBlocked(self, snake, snake_dir, left_direction_vector, right_direction_vector, box):
        front_blocked = 0
        left_blocked = 0
        right_blocked = 0
        snake_ = snake.copy()

        next_pos = snake_[0] + snake_dir
        #     next_pos = snake_[0]
        next_pos = next_pos.tolist()

        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            front_blocked = 1

        next_pos = snake_[0] + left_direction_vector
        next_pos = next_pos.tolist()
        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            left_blocked = 1

        next_pos = snake_[0] + right_direction_vector
        next_pos = next_pos.tolist()
        if next_pos[0] in [box[0][0], box[1][0]] or next_pos[1] in [box[0][1], box[1][1]] or next_pos in snake_:
            right_blocked = 1

        return left_blocked, front_blocked, right_blocked

    def dir_correction(self, direction, directionVect, left_blocked, front_blocked, right_blocked):

        if direction == [0, 1, 0]:
            if front_blocked:
                direction = [0, 0, 1]
                if right_blocked:
                    direction = [1, 0, 0]
                    
        elif direction == [0, 0, 1]: 
            if right_blocked:    
                direction = [0, 1, 0]   
                if front_blocked:
                    direction = [1, 0, 0]

        elif direction == [1, 0, 0]: 
            if left_blocked:    
                direction = [0, 1, 0]     
                if front_blocked:
                    direction = [0, 0, 1]

        return direction

    def init_render(self):
        stdscr = curses.initscr()
        curses.curs_set(0)  # stop cursor from blink
        stdscr.nodelay(1)  # getch method will not block the code anymore
        stdscr.clear()
        textpad.rectangle(stdscr, self.box[0][0], self.box[0][1], self.box[1][0], self.box[1][1])
        stdscr.refresh()
        #         stdscr.timeout(150)  # waits 150 ms till next loop
        #         time.sleep(3)
        return stdscr

    def render(self, stdscr, box, snake_, apple_):
        stdscr.clear()
        textpad.rectangle(stdscr, self.box[0][0], self.box[0][1], self.box[1][0], self.box[1][1])
        for s in snake_:
            stdscr.addstr(s[0], s[1], "O")
        stdscr.addstr(apple_[0], apple_[1], "*")
        stdscr.refresh()

    def quit_render(self, stdscr):
        curses.curs_set(1)
        stdscr.nodelay(0)
        curses.endwin()

    def NN(self, model_, input_vect):
        predictions = model_.predict(input_vect)
        return predictions
