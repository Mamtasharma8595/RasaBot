import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from apscheduler.schedulers.background import BackgroundScheduler
import pickle
import os
import mysql.connector
import pandas as pd
import logging
df = pd.DataFrame()
def fetch_data_from_db() -> pd.DataFrame:
    """Fetch data from the database and return as a DataFrame."""
    global df
    try:
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
                inventory.Activation_Date,
                inventory.IOrderId
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
                AND inventory.PurchaseDate BETWEEN '2022-11-01' AND '2024-11-28'  
            ORDER BY 
                inventory.PurchaseDate DESC;
        """

        # Connect to the MySQL database and fetch data into a DataFrame
        connection = mysql.connector.connect(
            host="34.42.98.10",       
            user="clayerp",   
            password="6z^*V2M9Y(/+", 
            database="esim_local" 
        )

        # Fetch the data into a DataFrame
        new_data = pd.read_sql(query, connection)

        # df = pd.read_sql(query, connection)

        # # Close the connection
        connection.close()

        logging.info("Data successfully loaded from the database.")
        
        # df = df.drop_duplicates()
        # df = df.replace({'\\N': np.nan, 'undefined': np.nan, 'null': np.nan})
        # df['SellingPrice'] = pd.to_numeric(df['SellingPrice'], errors='coerce').fillna(0).astype(int)
        # df['CompanyBuyingPrice'] = pd.to_numeric(df['CompanyBuyingPrice'], errors='coerce').fillna(0).astype(int)

        # # Convert 'PurchaseDate' to datetime format and then format it as required
        # df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        # #df['PurchaseDate'] = df['PurchaseDate'].dt.date
        # df['TravelDate'] = pd.to_datetime(df['TravelDate'], errors='coerce')
        # return df
        new_data = new_data.replace({'\\N': np.nan, 'undefined': np.nan, 'null': np.nan})
        new_data['SellingPrice'] = pd.to_numeric(new_data['SellingPrice'], errors='coerce').fillna(0).astype(int)
        new_data['CompanyBuyingPrice'] = pd.to_numeric(new_data['CompanyBuyingPrice'], errors='coerce').fillna(0).astype(int)
        new_data['PurchaseDate'] = pd.to_datetime(new_data['PurchaseDate'], errors='coerce')
        new_data['TravelDate'] = pd.to_datetime(new_data['TravelDate'], errors='coerce')

        # If new data is fetched, append it to the global DataFrame and remove duplicates
        if not new_data.empty:
            df = pd.concat([df, new_data], ignore_index=True).drop_duplicates()
            logging.info("New data successfully fetched and stored in memory.")
        else:
            logging.info("No new data found.")
        
        return df

    except mysql.connector.Error as e:
        logging.error(f"Database error: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

def schedule_data_update():
    """Schedule the data update job to run at midnight."""
    scheduler = BackgroundScheduler()
    scheduler.add_job(fetch_and_update_data, 'cron', hour=0, minute=0,second=0)
    scheduler.start()

def fetch_and_update_data():
    """Fetch and update the global DataFrame with new data."""
    global df
    last_fetched_date = get_last_fetched_date()
    logging.info("Starting data update...")

    # Fetch new data based on the last fetched date
    df = fetch_data_from_db(last_fetched_date)

def get_last_fetched_date() -> pd.Timestamp:
    """Retrieve the last fetched date from the DataFrame."""
    if not df.empty:
        return df['PurchaseDate'].max()
    return None

# Start the scheduler when utils.py is imported
schedule_data_update()
