from db.db_helpers import db_helpers as DB

db = DB()
DB.push_last_activity(db)
weight_change = 225

#DB.push(db, 1, weight_change)
#DB.push(db, 2, weight_change)
#DB.remove(db,no,weight_change)
