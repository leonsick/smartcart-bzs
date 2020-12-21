import boto3
from secrets import Secrets
# from ..items import item as item_helper
import time
from datetime import datetime


# AWS credentials

class db_helpers(object):
    def __init__(self):
        self.db = boto3.resource('dynamodb', region_name='eu-central-1', aws_access_key_id=Secrets.get_aws_access_key(),
                                 aws_secret_access_key=Secrets.get_aws_secret_access_key())
        self.cart_id = Secrets.get_cart_id()
        self.table = self.db.Table('cart_' + self.cart_id)
        self.products = self.db.Table('product_dataset')
        self.cart_info = self.db.Table('cart_info')

    '''
    Create a new db entry for a new item
        :param item: String of new item
        :param added_weight: Float with weight added to scale
    '''

    # TODO: ggf. abfrage der Produktdatenbank in separaten Methoden auslagern

    def push(self, product_number, weight_change):
        if product_number == 0:
            return  # nothing detected

        product = self.call_product_info(product_number, weight_change)
        # TODO if
        product_check = self.product_check(product[1])
        if not product_check[0]:
            try:
                self.table.put_item(
                    Item={
                        'item_number': str(int(time.time())),
                        'price_per_unit': str(product[0]),
                        'product': product[1],
                        'qty': str(product[2]),
                        'unit': product[3]
                    }
                )
                print('Item was successfully added to db!')
            except:
                print('Failed...try again')
        elif product_check[0]:
            new_qty = str(int(product[2]) + int(product_check[1]))
            response = self.table.update_item(
                Key={
                    'item_number': product_check[2]
                },
                UpdateExpression='SET qty = :new_qty',
                ExpressionAttributeValues={
                    ':new_qty': new_qty
                },
                ReturnValues="UPDATED_NEW"
            )
        pass

    def remove(self, product_number, weight_change):
        if product_number == 0:
            return  # nothing detected

        product = self.call_product_info(product_number, weight_change)
        product_check = self.product_check(product[1])
        if int(product_check[1]) <= abs(int(product[2])):
            try:
                self.table.delete_item(
                    Key={
                        'item_number': product_check[2]
                    }
                )
                print('Item was successfully deleted to db!')
            except:
                print('Failed...try again')
        elif int(product_check[1]) > abs(int(product[2])):
            new_qty = str(int(product_check[1]) - abs(int(product[2])))
            response = self.table.update_item(
                Key={
                    'item_number': product_check[2]
                },
                UpdateExpression='SET qty = :new_qty',
                ExpressionAttributeValues={
                    ':new_qty': new_qty
                },
                ReturnValues="UPDATED_NEW"
            )
        pass

    def call_product_info(self, product_number, weight_change):
        if product_number == 0:
            return
        name = None
        price_per_piece = None
        unit = None
        weight_per_piece = None
        products = self.products.scan()
        for product in products['Items']:
            if product['product_number'] == product_number:
                name = product['name']
                price_per_piece = product['price_per_piece']
                unit = product['unit']
                weight_per_piece = product['weight_per_piece']
        qty = int(round((float(weight_change) / float(weight_per_piece)), 0))  # rounding
        return price_per_piece, name, qty, unit

    def product_check(self, search_product):
        products = self.table.scan()
        for product in products['Items']:
            if product['product'] == search_product:
                qty = product['qty']
                item_number = product['item_number']
                return True, qty, item_number
        return False, 0, '0'

    def push_last_activity(self):
        current_time = str(datetime.utcnow())
        response = self.cart_info.update_item(
            Key={
                'cart_id': self.cart_id
            },
            UpdateExpression='SET last_activity = :new_last_activity',
            ExpressionAttributeValues={
                ':new_last_activity': current_time
            },
            ReturnValues="UPDATED_NEW"
        )