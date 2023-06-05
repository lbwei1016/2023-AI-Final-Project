from PIL import Image

def merge_mask(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    # img1 = img1.convert("RGB")
    # img2 = img2.convert("RGB")

    data1 = list(img1.getdata())
    data2 = list(img2.getdata())

    r, w = img1.size
    length = r * w

    # WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    # merge = [WHITE for i in range(length)]
    merge = [BLACK for i in range(length)]

    for i in range(length):
        # print(data1[i])
        # if data1[i] != WHITE:
        #     merge[i] = data1[i]
        # elif data2[i] != WHITE:
        #     merge[i] = data2[i]
        # else: merge[i] = WHITE

        if data1[i] != BLACK:
            merge[i] = data1[i]
        elif data2[i] != BLACK:
            merge[i] = data2[i]
        # else: merge[i] = BLACK

        # if type(merge[i]) is not tuple:
            # print(f"data1: {data1[i]}; data2: {data2[i]}")

    # print(merge)

    # img3 = Image.new("RGB", (r, w))
    # img3.putdata(merge)
    # img3.save("merged.png")

    return merge

    # a = [(1, 2), (3, 4)]
    # b = [(9, 8), (7, 6)]

    # a[0] = b[1]
    # print(a)