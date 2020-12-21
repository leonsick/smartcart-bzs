### How to authenticate a SmartCart

On each SmartCart, you will find a QR code and a 5 digit number.
To activate the SmartCarts online interface, please follow the these steps:

1. Scan the QR code on the cart
2. Enter the 5 digit code on the website
3. You're good to go! Your shopping tour can begin


#### Technical steps
- QR code is printed on cart, code is generated ahead of time
- Scanning of QR code transfers cartid to cloud: xxx.com/auth/cartid/<u>12345</u>
- Auth webpage needs to feature form that takes in the 5 digit code
- DB then has to be checked if cartid = 12345 matches the 5 digit code
- If **true**: Forward to shopping interface for cartid = 12345

Addition: Maybe use session ids