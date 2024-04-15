import sys
from pprint import pprint
from collections import defaultdict

def ex01():
    if len(sys.argv) != 2:
        raise Exception("Insert as argument the filename!")
    filename = sys.argv[1]

    with open(filename, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]
    lines = [[float(e) if e.replace(".","").isnumeric() else e for e in line] for line in lines]
    max_min = [(max(line[3:]), min(line[3:])) for line in lines]

    #genexp, more memory efficeint (I don't need the actual list of returned values), but I don't know why, if I put round brackets it doesn't remove the elements inside
    [lines[i].remove(max_min[i][0]) for i in range(len(lines))]
    [lines[i].remove(max_min[i][1]) for i in range(len(lines))]

    for line in lines:
        line[3:] = [sum(line[3:])]

    lines.sort(key=lambda l: l[3], reverse=True)

    print("final ranking")
    for i in range(3):
        print(f"{i+1}: {lines[i][0]} {lines[i][1]} - Score: {lines[i][3]}")

    country_dict = defaultdict(float)

    for line in lines:
        country_dict[line[2]] += line[3]

    max_key = max(country_dict, key=country_dict.get)
    print("best country")
    print(f"{max_key} - Total score: {country_dict[max_key]}")

def ex02():
    if len(sys.argv) != 4:
        raise Exception("Too few arguments!")
    
    if sys.argv[2] == "-b" or sys.argv[2] == "-l":
        with open(sys.argv[1]) as f:
            lines = f.readlines()
        # potrei convertire ad intero se mi sembra comodo
        lines = [line.strip().split() for line in lines]
    else: raise Exception("Invalid argument!")

    if sys.argv[2] == "-b":
        busId = sys.argv[3]
        lines = [line for line in lines if line[0] == busId]
        lines.sort(key=lambda e: e[-1])
        lines = [line[2:4] for line in lines]
        d = 0
        for i in range(len(lines)-1):
            d += ((float(lines[i][0])-float(lines[i+1][0]))**2 + (float(lines[i][1])-float(lines[i+1][1]))**2)**0.5
        print(f"{busId} - Total Distance: {d}")

    elif sys.argv[2] == "-l":
        lineId = sys.argv[3]
        lines = [line for line in lines if line[1] == lineId]
        if len(lines) == 0:
            print(f"{lineId} - Avg Speed: 0.0")
            return
        lines.sort(key=lambda e: e[-1])
        busIds = set([line[0] for line in lines])

        d = 0
        t = 0
        for busId in busIds:
            print(busId)
            f = [line for line in lines if line[0] == busId]
            pprint(f)
            for i in range(len(f)-1):
                print(f"index: {i}")
                d += ((float(f[i][2])-float(f[i+1][2]))**2 + (float(f[i][3])-float(f[i+1][3]))**2)**0.5
                t += abs((float(f[i][-1])-float(f[i+1][-1])))

        avg = d/t
        print(f"{lineId} - Avg Speed: {avg}")

def ex03():
    if len(sys.argv) != 2:
        raise Exception("Too few arguments!")
    
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    #NUMBER OF BIRTH FOR CITY
    print("Birth per city:")
    cities = set([line[2] for line in lines])
    for city in cities:
        f = [line for line in lines if line[2] == city]
        cities_num = len(f)
        print(f"{city}: {len(f)}")

    #NUMBER OF BIRTH FOR EACH MONTH
    print("Birth per month:")
    months = set([line[-1].split("/")[1] for line in lines])
    for month in months:
        f = [line for line in lines if line[-1].split("/")[1] == month]
        print(f"{month}: {len(f)}")

    #AVERAGE NUMBER OF BIRTH PER CITY
    print(f"Average # of birth: {len(lines)/cities_num}")
    
def ex04():
    if len(sys.argv) != 2:
        raise Exception("Error in arguments formatting!")
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    print("Available copies: ")
    isnbs = set([line[0] for line in lines])
    for isnb in isnbs:
        bought_lines = [line for line in lines if (line[0] == isnb) and (line[1] == "B")]
        sold_lines = [line for line in lines if (line[0] == isnb) and (line[1] == "S")]
        print(f"{isnb}: {sum([int(line[-2]) for line in bought_lines])-sum([int(line[-2]) for line in sold_lines])}")
        
    print("Sold books per month:")
    mys = set(["/".join(line[-3].split("/")[1:]) for line in lines])
    for my in mys:
        filtered_lines = [line for line in lines if ("/".join(line[-3].split("/")[1:]) == my) and (line[1] == "S")]
        m = my.split("/")[0]
        y = my.split("/")[1]
        num_sold_books = sum([int(line[-2]) for line in filtered_lines])
        if(num_sold_books != 0):
            print(f"{m}, {y}: {num_sold_books}")

    # NOT WORKING - I DON'T CARE
    print("Gain per book:")
    isnbs = set([line[0] for line in lines])
    for isnb in isnbs:
        filtered_lines = [line for line in lines if line[0] == isnb]
        sold_lines = [float(line[-1]) for line in lines if line[1] == "S"]
        sell_price = sum(sold_lines)/len(sold_lines)
        buy_lines = [float(line[-1]) for line in lines if line[1] == "B"]
        buy_price = sum(buy_lines)/len(buy_lines)
        print(f"{isnb}: {sell_price-(buy_price*len(sold_lines))} (avg {buy_price}, sold {len(sold_lines)})")

def ex05():
    if len(sys.argv) != 2:
        raise Exception("Wrong arguments number!")
    with open(sys.argv[1]) as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines]

    matrix  = [[0.0 for _ in range(7)] for _ in range(7)]
    for line in lines:
        x_sl = int(line[0])
        y_sl = int(line[1])
        for x, row in enumerate(matrix):
            for y, _ in enumerate(row):
                if x == x_sl and y == y_sl:
                    matrix[x][y] += 1.0
                elif (x >= x_sl-1 and x <= x_sl+1) and (y >= y_sl-1 and y <= y_sl+1):
                    matrix[x][y] += 1/2
                else:
                    matrix[x][y] += 1/5

    for row in matrix:
        for e in row:
            print(f"{e:.2f}", end=" ")
        print()


if __name__=="__main__":
    #ex01()
    #ex02()
    #ex03()
    #ex04()
    ex05()