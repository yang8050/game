import cv2
import numpy as np
from random import choice
import copy
from PIL import Image, ImageDraw, ImageFont

# 定义游戏界面 三通道 所有像素点的各通道数值赋0 类型为uint8
interface = np.ones([20, 10, 3], dtype=np.uint8)
interface[:, :] = [105, 139, 105]
# 控制俄罗斯方块下降的速度
speed = 0.5
# 退出
quit = False
# 放置
place = False
# 下落
drop = False
# 交换
switch = False
# 标志
flag = 0
# 分数
score = 0
# 下一个方块
# 返回列表、元组、或字符串的随机项
next_block = choice(["I", "o", "L", "J", "S", "Z", "T"])
# 持有方块
held_block = ""


def get_block(block):
    """
    不同颜色的形状 七种
    :param block: 形状
    :return: 形状，颜色
    """
    if block == "I":
        coordinate = np.array([[0, 3], [0, 4], [0, 5], [0, 6]])
        color = [255, 218, 185]
    elif block == "o":
        coordinate = np.array([[0, 4], [0, 5], [1, 4], [1, 5]])
        color = [191, 239, 255]
    elif block == "L":
        coordinate = np.array([[1, 3], [1, 4], [1, 5], [0, 5]])
        color = [202, 255, 112]
    elif block == "J":
        coordinate = np.array([[1, 3], [1, 4], [1, 5], [0, 3]])
        color = [255, 246, 143]
    elif block == "S":
        coordinate = np.array([[1, 3], [1, 4], [0, 4], [0, 5]])
        color = [216, 191, 216]
    elif block == "Z":
        coordinate = np.array([[1, 5], [1, 4], [0, 3], [0, 4]])
        color = [240, 255, 240]
    else:
        coordinate = np.array([[1, 3], [1, 4], [1, 5], [0, 4]])
        color = [255, 160, 122]

    return coordinate, color

def display(interface, coordinate, color, next_get_block, held_get_block, score, speed):
    # 背景条 竖条 高20宽1 3通道 所有像素点各通道赋值0
    frame = np.zeros([20, 1, 3], dtype=np.uint8)
    # 法一
    frame[:, :, 0] = np.ones([20, 1]) * 193
    frame[:, :, 1] = np.ones([20, 1]) * 255
    frame[:, :, 2] = np.ones([20, 1]) * 193

    # 背景条 横条 高1宽34 3通道 所有像素点各通道赋值0
    frame_ = np.ones([1, 23, 3], dtype=np.uint8)
    # 和竖条定义颜色的方法不一样 法二
    frame_[:, :] = [193, 255, 193]

    # 浅拷贝游戏界面
    right = copy.copy(interface)
    # right[:, :] = [0, 0, 0]
    # 游戏界面填充方块的颜色
    right[coordinate[:, 0], coordinate[:, 1]] = color

    # 左边游戏界面
    left = np.zeros([20, 10, 3], dtype=np.uint8)
    left[:, :] = [105, 139, 105]
    left[next_get_block[0][:, 0] + 2, next_get_block[0][:, 1]] = next_get_block[1]
    left[held_get_block[0][:, 0] + 11, held_get_block[0][:, 1] + 3] = held_get_block[1]

    # 拼接
    # 按第二维度进行拼接
    entire = np.concatenate((frame, left, frame, right, frame), 1)
    # 按第一维度进行拼接
    entire = np.concatenate((frame_, entire, frame_), 0)
    # 矢量放大 先第一维度放大 再第二维度放大
    entire = entire.repeat(35, 0).repeat(35, 1)
    # 放置分数
    entire = cv2.putText(entire, str(score), (190, 240), cv2.FONT_HERSHEY_DUPLEX, 1, [193, 255, 193], 2)
    # 放置说明
    entire = cv2AddChineseText(entire, "A - 左移", (75, 300), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "D - 右移", (75, 350), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "S - 快速下移", (75, 400), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "W - 直接降落", (75, 450), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "J - 向左旋转", (75, 500), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "L - 向右旋转", (75, 550), (193, 255, 193), 20)
    entire = cv2AddChineseText(entire, "I - 交换", (75, 600), (193, 255, 193), 20)

    cv2.imshow("Tetris", entire)
    # 等待返回按键信息
    key = cv2.waitKey(int(1000 / speed))

    return key

# 添加中文
def cv2AddChineseText(img, text, position, textColor, textSize):
    if (isinstance(img, np.ndarray)):    # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    while not quit:
        # 是否想要交换保留块与当前块
        if switch:
            # 交换保留块与当前块
            held_block, current_block = current_block, held_block
            switch = False
        else:
            # 生成下一块并更新当前一块
            current_block = next_block
            next_block = choice(["I", "o", "L", "J", "S", "Z", "T"])

        if flag > 0:
            flag -= 1

        # 是否持有方块 无则显示背景色
        if held_block == "":
            held_get_block = np.array([[0, 0]]), [105, 139, 105]
        else:
            held_get_block = get_block(held_block)
        # 获取下一个俄罗斯方块
        next_get_block = get_block(next_block)
        # 获取当前俄罗斯方块
        coordinate, color = get_block(current_block)
        if current_block == "I":
            top_left = [-2, 3]

        # 前两行区域有其它方块 游戏结束
        if not np.all(interface[coordinate[:, 0], coordinate[:, 1]] == [105, 139, 105]):
            break

        while True:
            # 显示背景板游戏界面、获取按键信息
            key = display(interface, coordinate, color, next_get_block, held_get_block, score, speed)
            # 创建位置的副本
            dummy = coordinate.copy()

            if key == ord("a"):
                # 如果块不靠左墙，则向左移动块
                if np.min(coordinate[:, 1]) > 0:
                    coordinate[:, 1] -= 1
                if current_block == "I":
                    top_left[1] -= 1

            elif key == ord("d"):
                # 如果它不靠在右墙上，则将其向右移动
                if np.max(coordinate[:, 1]) < 9:
                    coordinate[:, 1] += 1
                    if current_block == "I":
                        top_left[1] += 1

            elif key == ord("j") or key == ord("l"):
                # 旋转
                # arr 是旋转的附近点的数组，pov 是 arr 内块的位置索引

                if current_block != "I" and current_block != "O":
                    if coordinate[1, 1] > 0 and coordinate[1, 1] < 9:
                        # 构建三维数组
                        arr = coordinate[1] - 1 + np.array([[[x, y] for y in range(3)] for x in range(3)])
                        # coordinate[1]需是方块中间 若是第一排 则 coordinate[1] - coordinate + 1
                        # 若固定coordinate[1]是方块第二排中间项 则 coordinate - coordinate[1] + 1
                        # 由于有两个方块没有第一排中间项 还是固定为第二排中间块
                        # pov = coordinate - coordinate[1] + 1
                        pov = np.array(
                            [np.where(np.logical_and(arr[:, :, 0] == pos[0], arr[:, :, 1] == pos[1])) for pos in
                             coordinate])
                        # 关掉一个轴 二维度
                        pov = np.array([k[0] for k in np.swapaxes(pov, 1, 2)])

                elif current_block == "I":
                    arr = top_left + np.array([[[x, y] for y in range(4)] for x in range(4)])
                    pov = np.array(
                        [np.where(np.logical_and(arr[:, :, 0] == pos[0], arr[:, :, 1] == pos[1])) for pos in coordinate])
                    # 关掉一个轴 二维度
                    pov = np.array([k[0] for k in np.swapaxes(pov, 1, 2)])

                # 转阵列并将块重新定位到它现在的位置

                if current_block != "O":
                    if key == ord("j"):
                        arr = np.rot90(arr, -1)
                    else:
                        arr = np.rot90(arr)
                    coordinate = arr[pov[:, 0], pov[:, 1]]

            elif key == ord("w"):
                # 直接下降
                drop = True
            elif key == ord("i"):
                # 退出循环并告诉程序交换保留块和当前块
                if flag == 0:
                    if held_block == "":
                        held_block = current_block
                    else:
                        switch = True
                    flag = 2
                    break
            # 退出 8-Backspace 27-esc
            elif key == 8 or key == 27:
                quit = True
                break

            # 检查块是否与其他块重叠或是否在棋盘外，如果是，则将位置更改为发生任何事情之前的位置
            if np.max(coordinate[:, 0]) < 20 and np.min(coordinate[:, 0]) >= 0:
                if not (current_block == "I" and (np.max(coordinate[:, 1]) >= 10 or np.min(coordinate[:, 1]) < 0)):
                    if not np.all(interface[coordinate[:, 0], coordinate[:, 1]] == [105, 139, 105]):
                        coordinate = dummy.copy()
                else:
                    coordinate = dummy.copy()
            else:
                coordinate = dummy.copy()

            # 是否直接下落
            if drop:
                while not place:
                    if np.max(coordinate[:, 0]) != 19:
                        # 检查块是否停靠
                        for pos in coordinate:
                            # 判断两个数组是否相等
                            if not np.array_equal(interface[pos[0] + 1, pos[1]], [105, 139, 105]):
                                place = True
                                break
                    else:
                        # 在棋盘底部，则直接放置
                        place = True

                    if place:
                        break

                    # 继续下降并检查何时需要放置块

                    coordinate[:, 0] += 1
                    if current_block == "I":
                        top_left[0] += 1

                drop = False

            else:
                # 检查是否需要放置块
                if np.max(coordinate[:, 0]) != 19:
                    for pos in coordinate:
                        if not np.array_equal(interface[pos[0] + 1, pos[1]], [105, 139, 105]):
                            place = True
                            break
                else:
                    place = True

            if place:
                # 将块放在棋盘上的位置
                for pos in coordinate:
                    interface[tuple(pos)] = color

                # 将place重置为 False
                place = False
                break

            # 向下移动 1
            coordinate[:, 0] += 1
            if current_block == "I":
                top_left[0] += 1

        # 清除行并计算已清除的行数并更新分数

        lines = 0

        for line in range(20):
            if np.all([np.any(pos != [105, 139, 105]) for pos in interface[line]]):
                lines += 1
                interface[1:line + 1] = interface[:line]

        if lines == 1:
            score += 1
        elif lines == 2:
            score += 2
        elif lines == 3:
            score += 3
        elif lines == 4:
            score += 4
        elif lines == 5:
            score += 5
        elif lines == 6:
            score += 6

