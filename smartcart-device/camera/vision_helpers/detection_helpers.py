from items import item

apple = item("Apple", price_per_unit=1.00, weight_per_unit=0.1, priced_by_weight=True)
banana = item("Banana", price_per_unit=1.99, weight_per_unit=0.1, priced_by_weight=True)
beer = item("Beer", price_per_unit=0.50, weight_per_unit=0.5, priced_by_weight=False)
paprika = item("Paprika", price_per_unit=0.60, weight_per_unit=0.3, priced_by_weight=True)
broccoli = item("Broccoli", price_per_unit=1.20, weight_per_unit=1.50, priced_by_weight=True)
cabbage = item("Cabbage", price_per_unit=0.30, weight_per_unit=0.5, priced_by_weight=True)
cheese = item("Cheese", price_per_unit=2.00, weight_per_unit=0.8, priced_by_weight=False)
guacamole = item("Guacamole", price_per_unit= 3.00, weight_per_unit= 0.5, priced_by_weight=False)
milk = item("Milk", price_per_unit=0.5, weight_per_unit=1.0, priced_by_weight=False)
radish = item("Radish", price_per_unit=0.5, weight_per_unit=0.8, priced_by_weight=True)
zucchini = item("Zucchini", price_per_unit=0.6, weight_per_unit=0.7, priced_by_weight=True)

sortiment = [apple, banana, beer, paprika, broccoli, cabbage, cheese, guacamole, milk, radish, zucchini]

def object_is_detected():
    pass