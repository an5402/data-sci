text = open("GOOG_hist.txt")
line = text.readline()
raw_data = []
data = []

# Initial clean-up
while line != "":
    raw_data.append(line)
    line = text.readline()

for item in raw_data:
    new_item = float(item.replace("\n", ""))
    data.append(new_item)

data = list(reversed(data))


class MovingAverageExperiment:
    def __init__(self, short_time, over_spread, under_spread, cap, stock_number):
        # TODO allow parameters to vary, find optimal
        self.short_time = short_time
        self.over_spread = over_spread
        self.under_spread = under_spread
        self.cap = cap
        self.stock_number = stock_number

    def __str__(self):
        return "Portfolio with: " + str(self.cap) + " in free capital and: " + str(self.stock_number) + "GOOG stocks."

    def calculate_long_average(self, date):
        i = 0
        stock_sum = 0
        while i <= date:
            stock_sum += data[i]
            i += 1
        return stock_sum / i

    def calculate_short_average(self, date):
        i = date - self.short_time
        stock_sum = 0
        while i <= date:
            stock_sum += data[i]
            i += 1
        return stock_sum / (date - self.short_time)

    def make_trade(self, date):
        if self.calculate_long_average(date) > self.calculate_short_average(date):
            self.buy_stock(date)
        else:
            self.sell_stock(date)

    def buy_stock(self, date):
        if self.cap >= data[date]:
            self.cap -= data[date]
            self.stock_number += 1

    def sell_stock(self, date):
        self.cap += data[date]
        self.stock_number -= 1

    def check_rule(self, date):
        if self.calculate_long_average(date) - self.calculate_short_average(date) >= self.under_spread:
            self.make_trade(date)
        elif self.calculate_short_average(date) - self.calculate_long_average(date) >= self.over_spread:
            self.make_trade(date)

    def run_experiment(self):
        date = 10  # TODO fix issues with a short start date
        while date < len(data):  # TODO fix issues with date not properly working
            self.check_rule(date)
            date += 1
        if self.stock_number > 0:
            self.cap += self.stock_number * data[-1]
            self.stock_number = 0
        return self.cap  # TODO make a more informative summary

e = MovingAverageExperiment(5, 100, 100, 10000, 0)
print(e.run_experiment())

