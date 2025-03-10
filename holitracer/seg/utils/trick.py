class EarlyStopping(object):
    def __init__(self, criterion: float = 0.0, patience: int = 7):
        """
        连续 patience 轮 criterion 指标不增加
        """
        self.EarlyStopping = False
        self.patience = patience
        self.count = 0
        self.criterion = criterion

    def CheckStopping(self, new_criterion):
        flag = False
        if new_criterion < self.criterion:
            self.count += 1
            print(f"EarlyStopping counter: {self.count} out of {self.patience}")
        else:
            print(f"EarlyStopping criterion : {self.criterion} => {new_criterion}")
            self.criterion = new_criterion
            self.count = 0
            flag = True
        if self.count >= self.patience:
            self.EarlyStopping = True
        return flag