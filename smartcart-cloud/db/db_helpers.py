import json
import boto3
from secrets_cloud import Secrets_cloud
import time
from traceback import print_exc
from datetime import datetime

# AWS credentials
aws_access_key = Secrets_cloud.get_aws_access_key() #os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = Secrets_cloud.get_aws_secret_access_key()  #os.environ['AWS_SECRET_ACCESS_KEY']
dynamo = boto3.resource('dynamodb', region_name='eu-central-1', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_access_key)


def query_for_auth(code, cart_id):
    cart_id_from_db = db.search_for(code, value="cart_id")

    if not cart_id_from_db:
        return False

    if cart_id_from_db == cart_id:
        return True


def get_cart_code(cart_id):
    table = dynamo.Table('cart_info')
    items = table.scan()
    cart_code = None
    for items in items['Items']:
        if items['cart_id'] == cart_id:
            cart_code = items['cart_code']
    return cart_code


def get_cart_session(cart_id):
    table = dynamo.Table('cart_info')
    items = table.scan()
    cart_session = None
    for items in items['Items']:
        if items['cart_id'] == cart_id:
            cart_session = items['cart_session']
    return cart_session


def set_cart_session(cart_id, new_cart_session):
    table = dynamo.Table('cart_info')
    response = table.update_item(
        Key={
            'cart_id': cart_id
        },
        UpdateExpression='SET cart_session = :new_cart_session',
        ExpressionAttributeValues={
            ':new_cart_session': new_cart_session
        },
        ReturnValues="UPDATED_NEW"
    )


def get_cart_db(cart_id):
    print("Hier kommt die CartId:" +cart_id)
    table = dynamo.Table('cart_info')
    items = table.scan()
    cart_db = None
    for items in items['Items']:
        if items['cart_id'] == cart_id:
            cart_db = items['cart_db']
    return cart_db


def get_products(cart_id):
    cart_db = get_cart_db(cart_id)
    print(cart_db)
    table = dynamo.Table(cart_db)
    items = table.scan()
    products = []
    for items in items['Items']:
        products.append(items)
    return sort(products)

def get_archive_id():
    table = dynamo.Table('archive')
    contents = table.scan()
    current_max_id = int(0)
    for row in contents['Items']:
        print(contents)
        if int(row['id']) > current_max_id:
            current_max_id = int(row['id'])
    return current_max_id

def push_to_archive(cart_id, cart_session):
    print("CartID push to archive:" + cart_id)
    print("Cartsession push to archive:" + cart_session)
    table_Archive = dynamo.Table('archive')
    archive_id = get_archive_id()

    cart_db = get_cart_db(cart_id)
    print(cart_db)
    table = dynamo.Table(cart_db)
    items = table.scan()
    for product in items['Items']:
        archive_id = archive_id +1
        try:
            table_Archive.put_item(
                Item={
                    'id': archive_id,
                    'cart': cart_id,
                    'cart_session': cart_session,
                    'time': str(datetime.now()),
                    'item_number': str(product['item_number']),
                    'price_per_unit':str(product['price_per_unit']),
                    'product': product['product'],
                    'qty': str(product['qty']),
                    'unit': product['unit']
                }
            )
            print('Item was successfully added to db!')
        except:
            print('Failed...try again')
            print(print_exc())
    empty_cart(cart_db)

def empty_cart(cart_db):
    table = dynamo.Table(cart_db)

    scan = table.scan(
        ProjectionExpression='#k',
        ExpressionAttributeNames={
            '#k': 'item_number'
        }
    )

    with table.batch_writer() as batch:
        for each in scan['Items']:
            batch.delete_item(Key=each)

def sort(array, criteria='item_number'):
    """Sort the array by using quicksort."""

    less = []
    equal = []
    greater = []

    if len(array) > 1:
        pivot = array[0]
        pivot_compare = int(array[0][criteria])
        for x in array:
            x_compare = int(x[criteria])
            if x_compare > pivot_compare:
                less.append(x)
            elif x_compare == pivot_compare:
                equal.append(x)
            elif x_compare < pivot_compare:
                greater.append(x)
        # Don't forget to return something!
        return sort(less)+equal+sort(greater)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to handle the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array


def get_last_activity(cart_id):
    table = dynamo.Table('cart_info')
    items = table.scan()
    last_activity = None
    for items in items['Items']:
        if items['cart_id'] == cart_id:
            last_activity = items['last_activity']
            print('last activity found!')
            print(last_activity)
    return last_activity

