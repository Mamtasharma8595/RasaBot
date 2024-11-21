# from rasa_sdk import Tracker
# from rasa_sdk.executor import CollectingDispatcher
# from rasa_sdk.types import DomainDict
# from actions.py import ActionCompareCountries  # Adjust import

# # Create a sample tracker with mock data for testing
# tracker = Tracker(
#     sender_id="test_user",
#     slots={},
#     latest_message={"text": "Compare sales between UK and USA in 2023"},
#     events=[],
#     paused=False,
#     followup_action=None,
#     active_loop=None,
#     latest_action_name=None,
# )

# # Instantiate the dispatcher
# dispatcher = CollectingDispatcher()

# # Define a dummy domain
# domain = {}

# # Instantiate and run your action
# action = ActionCompareCountries()
# action.run(dispatcher, tracker, domain)

# # Print the messages that would have been sent to the user
# for message in dispatcher.messages:
#     print(message["text"])"""
import mysql.connector
import pandas as pd
import logging

def fetch_data_from_db() -> pd.DataFrame:
    """Fetch data from the database and return as a DataFrame."""
    try:
        # SQL query to fetch sales data (same query as before)
        query = """
            SELECT 
                inventory.id,
                a.OperatorType,
                users.UserName,
                checkout.city,
                users.emailid AS Email, 
                checkout.country_name,
                inventory.PurchaseDate,
                DATE_FORMAT(inventory.PurchaseDate, '%r') AS `time`,   
                checkout.price AS SellingPrice,
                (SELECT planename FROM tbl_Plane WHERE P_id = inventory.planeid) AS PlanName,
                a.vaildity AS vaildity,
                (SELECT CountryName FROM tbl_reasonbycountry WHERE ID = a.country) AS countryname,
                (SELECT Name FROM tbl_region WHERE ID = a.region) AS regionname,
                (CASE 
                    WHEN (inventory.transcation IS NOT NULL OR Payment_method = 'Stripe') 
                    THEN 'stripe' 
                    ELSE 'paypal' 
                END) AS payment_gateway,
                checkout.source,
                checkout.Refsite,
                checkout.accounttype,
                checkout.CompanyBuyingPrice,
                checkout.TravelDate,
                inventory.Activation_Date
            FROM 
                tbl_Inventroy inventory
            LEFT JOIN 
                tbl_plane AS a ON a.P_id = inventory.planeid
            LEFT JOIN 
                User_Login users ON inventory.CustomerID = users.Customerid
            LEFT JOIN 
                Checkoutdata checkout ON checkout.guid = inventory.guid
            WHERE 
                inventory.status = 3 
                AND inventory.PurchaseDate BETWEEN '2022-11-01' AND '2024-11-01'  
            ORDER BY 
                inventory.PurchaseDate DESC;
        """

        # Connect to the MySQL database and fetch data into a DataFrame
        connection = mysql.connector.connect(
            host="34.42.98.10",       
            user="clayerp",   
            password="6z^*V2M9Y(/+", 
            database="esim" 
        )

        # Fetch the data into a DataFrame
        df = pd.read_sql(query, connection)

        # Close the connection
        connection.close()

        logging.info("Data successfully loaded from the database.")
        return df

    except mysql.connector.Error as e:
        logging.error(f"Database error: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error




# import mysql.connector
# import pandas as pd
# import logging

# # Global variable to hold the connection
# db_connection = None

# def get_db_connection() -> mysql.connector.MySQLConnection:
#     """Get an active connection to the database. Reconnect if the connection is closed."""
#     global db_connection
    
#     # Check if the connection is open
#     if db_connection is None or not db_connection.is_connected():
#         try:
#             # Establish a new connection if it's not open
#             db_connection = mysql.connector.connect(
#                 host="your_host",       # Replace with actual MySQL host
#                 user="your_username",   # Replace with actual MySQL username
#                 password="your_password", # Replace with actual MySQL password
#                 database="your_database_name" # Replace with actual database name
#             )
#             logging.info("Database connection established.")
#         except mysql.connector.Error as e:
#             logging.error(f"Error establishing connection: {e}")
#             return None  # Return None if connection can't be established
#     return db_connection

# def fetch_data_from_db() -> pd.DataFrame:
#     """Fetch data from the database and return as a DataFrame."""
#     try:
#         connection = get_db_connection()
#         if connection is None:
#             return pd.DataFrame()  # Return an empty DataFrame if connection fails

#         # SQL query to fetch sales data (same query as before)
#         query = """
#             SELECT 
#                 inventory.id,
#                 a.OperatorType,
#                 users.UserName,
#                 checkout.city,
#                 users.emailid AS Email, 
#                 checkout.country_name,
#                 inventory.PurchaseDate,
#                 DATE_FORMAT(inventory.PurchaseDate, '%r') AS `time`,   
#                 checkout.price AS SellingPrice,
#                 (SELECT planename FROM tbl_Plane WHERE P_id = inventory.planeid) AS PlanName,
#                 a.vaildity AS vaildity,
#                 (SELECT CountryName FROM tbl_reasonbycountry WHERE ID = a.country) AS countryname,
#                 (SELECT Name FROM tbl_region WHERE ID = a.region) AS regionname,
#                 (CASE 
#                     WHEN (inventory.transcation IS NOT NULL OR Payment_method = 'Stripe') 
#                     THEN 'stripe' 
#                     ELSE 'paypal' 
#                 END) AS payment_gateway,
#                 checkout.source,
#                 checkout.Refsite,
#                 checkout.accounttype,
#                 checkout.CompanyBuyingPrice,
#                 checkout.TravelDate,
#                 inventory.Activation_Date
#             FROM 
#                 tbl_Inventroy inventory
#             LEFT JOIN 
#                 tbl_plane AS a ON a.P_id = inventory.planeid
#             LEFT JOIN 
#                 User_Login users ON inventory.CustomerID = users.Customerid
#             LEFT JOIN 
#                 Checkoutdata checkout ON checkout.guid = inventory.guid
#             WHERE 
#                 inventory.status = 3 
#                 AND inventory.PurchaseDate BETWEEN '2024-10-01' AND '2024-11-01'  
#             ORDER BY 
#                 inventory.PurchaseDate DESC;
#         """

#         # Fetch the data into a DataFrame
#         df = pd.read_sql(query, connection)

#         # Close the connection
#         connection.close()
#         logging.info("Data successfully loaded from the database.")

#         return df

#     except mysql.connector.Error as e:
#         logging.error(f"Database error: {str(e)}")
#         return pd.DataFrame()  # Return an empty DataFrame in case of error
