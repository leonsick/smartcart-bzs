class item(object):
    def __init__(self, product, price_per_unit, qty, unit):
        self.product = product
        self.price_per_unit = price_per_unit
        self.qty = qty
        self.unit = unit

    def get_product(self):
        return self.product

    def get_price_per_unit(self):
        return self.price_per_unit

    def get_qty(self):
        return self.qty

    def get_unit(self):
        return self.unit
