from fuzzywuzzy import process
from typing import Any, Text, Dict, Tuple, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from dateutil.relativedelta import relativedelta
import mysql.connector
import logging
df = pd.DataFrame()
logging.basicConfig(level=logging.INFO)



def fetch_data_from_db() -> pd.DataFrame:
    """Fetch data from the database and return as a DataFrame."""
    global df
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
        # df = df.drop_duplicates()
        df = df.replace({'\\N': np.nan, 'undefined': np.nan, 'null': np.nan})
        df['SellingPrice'] = pd.to_numeric(df['SellingPrice'], errors='coerce').fillna(0).astype(int)
        df['CompanyBuyingPrice'] = pd.to_numeric(df['CompanyBuyingPrice'], errors='coerce').fillna(0).astype(int)

        # Convert 'PurchaseDate' to datetime format and then format it as required
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        #df['PurchaseDate'] = df['PurchaseDate'].dt.date
        df['TravelDate'] = pd.to_datetime(df['TravelDate'], errors='coerce')
        return df

    except mysql.connector.Error as e:
        logging.error(f"Database error: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

fetch_data_from_db()


months = { 'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12 }
months_reverse = {v: k for k, v in months.items()}

word_to_num = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
    'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
    'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12
}

def extract_months_from_text(text):
    # Regular expression to match numbers or words like '5', 'five', etc.
    pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b'
    
    # Search for the number of months in the text
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Extract the matched part
        month_str = match.group(1).lower()
        
        # Convert word-based numbers to digits, or just use the digit if it's already a number
        if month_str.isdigit():
            return int(month_str)
        else:
            return word_to_num.get(month_str, 0)  # Convert word to number, default to 0 if not found
    return 0

def get_month_range_from_text(text):
    # Extract the number of months
    num_months = extract_months_from_text(text)
    
    if num_months > 0:
        # Get the current date
        current_date = datetime.now()
        
        # Calculate the date X months ago
        past_date = current_date - relativedelta(months=num_months)
        
        # Return the range of months (past month/year to current month/year)
        month_range = [[past_date.month, past_date.year], [current_date.month, current_date.year]]
        return month_range, num_months
    else:
        return None
def extract_specific_date(user_message):
    # Regex pattern for extracting a specific date like "5 April 2024"
    date_pattern = [
        r'\b\d{4}-\d{2}-\d{2}\b',             # YYYY-MM-DD
        r'\b\d{1,2}[ -/]\d{1,2}[ -/]\d{4}\b', # DD-MM-YYYY or MM-DD-YYYY
        r'\b\d{1,2}(st|nd|rd|th)? [A-Za-z]+ \d{4}\b',  # DD Month YYYY
        r'\b[A-Za-z]+ \d{1,2}(st|nd|rd|th)? \d{4}\b',  # Month DD YYYY
        r'\b[A-Za-z]+ \d{1,2}(st|nd|rd|th)?\b'   # Month DD
    ]
    combined_pattern = r'|'.join(date_patterns)
    
    # Search for date matches in the user message
    matches = re.findall(combined_pattern, user_message, re.IGNORECASE)
    for match in matches:
        if len(match) == 4:  # For the DD Month YYYY and Month DD YYYY formats
            if match[0].isdigit():  # This is DD Month YYYY
                day, _, month_name, year = match
            else:  # This is Month DD YYYY
                month_name, day, _, year = match

            month = months_reverse[month_name.lower()]
            specific_date = pd.to_datetime(f"{year}-{month}-{day.zfill(2)}", format="%Y-%m-%d")
            return specific_date

        elif len(match) == 1:  # For the YYYY-MM-DD format
            specific_date = pd.to_datetime(match[0], errors='coerce')
            if pd.notna(specific_date):
                return specific_date
        
        elif len(match) == 2:  # For DD-MM-YYYY or MM-DD-YYYY formats
            day_month = match[0].split('-')
            if len(day_month) == 2:  # Either DD-MM or MM-DD
                day = day_month[0].zfill(2)
                month = day_month[1].zfill(2)
                year = match[1]
                specific_date = pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d")
                return specific_date

    return None

# def extract_months(text):
#     # Regular expression to find month names or abbreviations (case-insensitive)
#     pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
#     # Find all matches of month names (case-insensitive)
#     matches = re.findall(pattern, text, re.IGNORECASE)
    
#     # Convert matched month names to corresponding digits
#     month_digits = [months[match.lower()] for match in matches]
    
#     return month_digits

# def extract_date_range(text):
   
#     try:
#         # Define patterns for start_date and end_date
#         pattern = r"from\s+([\w\s,]+)\s+to\s+([\w\s,]+)|between\s+([\w\s,]+)\s+and\s+([\w\s,]+)|from\s+([\w\s,]+)\s+through\s+([\w\s,]+)"
#         match = re.search(pattern, text, re.IGNORECASE)
        
#         if match:
#             # Extract start_date and end_date
#             start_date_str = match.group(1) or match.group(3) or match.group(5)
#             end_date_str = match.group(2) or match.group(4) or match.group(6)
            
#             # Parse dates
#             start_date = pd.to_datetime(start_date_str, errors='coerce')
#             end_date = pd.to_datetime(end_date_str, errors='coerce')
            
#             # Validate parsed dates
#             if pd.isnull(start_date) or pd.isnull(end_date):
#                 return None, None, "Error: One or both dates could not be parsed. Please provide valid dates."
            
#             return start_date.date(), end_date.date(), None
        
#         return None, None, "Error: No valid date range found in the query."
    
#     except Exception as e:
#         return None, None, f"Error occurred while parsing date range: {str(e)}"

def extract_months(text):
    """Extracts month numbers from user input based on month names or numeric/ordinal representations."""
    
    # Regular expression to find month names or abbreviations (case-insensitive)
    month_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
    # Regular expression to find numeric months (1-12)
    numeric_pattern = r'\b(1[0-2]|[1-9])\b'
    
    # Regular expression to find ordinal months (first to twelfth)
    ordinal_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b'
    
    # Find all matches of month names
    matches = re.findall(month_pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits
    month_digits = [months[match.lower()] for match in matches]

    # Check for numeric months
    numeric_match = re.search(numeric_pattern, text)
    if numeric_match:
        month_digits.append(int(numeric_match.group(0)))

    # Check for ordinal months
    ordinal_match = re.search(ordinal_pattern, text)
    if ordinal_match:
        month_digits.append(word_to_num.get(ordinal_match.group(0), None))

    return list(set(month_digits))
def extract_today(text):
    # Regular expression to match the word 'today'
    pattern = r'\btoday\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)

def extract_last_day(text):
    pattern = r'\blast\sday\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)


def extract_years(text):
    # Regular expression to match years in YYYY format without capturing the century separately
    pattern = r'\b(?:19|20)\d{2}\b'
    
    # Find all matches of the pattern
    years = re.findall(pattern, text)
    
    return [int(year) for year in years]
    

def extract_month_year(text):    
    pattern = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:of\s*)?(\d{4})\b'
    
    # Find all matches of the pattern (month and year)
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits and pair them with the year as arrays
    month_year_pairs = [[months[month.lower()], int(year)] for month, year in matches]
    
    return month_year_pairs

def extract_quarters_from_text(text):
    # Regular expression to match quarter-related terms
    pattern = r'\b(?:quarter\s*(1|2|3|4)|q(1|2|3|4)|first|second|third|fourth)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        # Normalize the matched group into a quarter number
        if match.group(1):  # Matches "quarter 1", "quarter 2", etc.
            quarter = int(match.group(1))
        elif match.group(2):  # Matches "q1", "q2", etc.
            quarter = int(match.group(2))
        else:  # Matches "first", "second", "third", "fourth"
            quarter_name = match.group(0).lower()
            quarter_map = {
                'first': 1,
                'second': 2,
                'third': 3,
                'fourth': 4
            }
            quarter = quarter_map.get(quarter_name, 0)
        
        # Return the corresponding month range for the identified quarter
        quarters = {
            1: (1, 3),   # Q1: January to March
            2: (4, 6),   # Q2: April to June
            3: (7, 9),   # Q3: July to September
            4: (10, 12)  # Q4: October to December
        }
        return quarters.get(quarter)
    
    return None
# def extract_quarters_from_text(text, current_year=None):
#     pattern = r'\b(?:quarter\s*(1|2|3|4)|q(1|2|3|4)|first|second|third|fourth)\s*(\d{4})?\b'
    
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     if match:
#         # Normalize the matched group into a quarter number
#         if match.group(1):  # Matches "quarter 1", "quarter 2", etc.
#             quarter = int(match.group(1))
#         elif match.group(2):  # Matches "q1", "q2", etc.
#             quarter = int(match.group(2))
#         else:  # Matches "first", "second", "third", "fourth"
#             quarter_name = match.group(0).lower()
#             quarter_map = {
#                 'first': 1,
#                 'second': 2,
#                 'third': 3,
#                 'fourth': 4
#             }
#             quarter = quarter_map.get(quarter_name, 0)

#         # Extract the year from the query if provided; otherwise, use the current year
#         year = match.group(3) if match.group(3) else current_year

#         if not year:  # If no year is specified and it's not the current year, return None
#             return None
        
#         # Map quarter to month range
#         quarters = {
#             1: (1, 3),   # Q1: January to March
#             2: (4, 6),   # Q2: April to June
#             3: (7, 9),   # Q3: July to September
#             4: (10, 12)  # Q4: October to December
#         }

#         # Return the quarter's month range along with the year
#         start_month, end_month = quarters.get(quarter, (None, None))
        
#         if start_month and end_month:
#             return (year, start_month, end_month)

#     return None

# def extract_half_year_from_text(text):
#     # Regular expression to match variations of half-yearly references
#     pattern = r'\b(first|second|sec)\s+(half\s+year|half\s+yearly|half\s+year\s+report|half\s+year\s+analysis|half\s+yearly\s+summary)\b'
#     match = re.search(pattern, text, re.IGNORECASE)

#     if match:
#         half = match.group(1).lower()
#         if half in ['first']:
#             return (1, 6)  # January to June
#         elif half in ['second', 'sec']:
#             return (7, 12)  # July to December
#     return None


def extract_half_year_from_text(text):
    pattern = r'\b(first|second|sec|1st|2nd)\s+(half|half\s+year|half\s+yearly|half\s+year\s+report|half\s+year\s+analysis|half\s+yearly\s+summary)\b'
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        half = match.group(1).lower()
        
        # Determine the months based on the half-year mentioned
        if half in ['first', '1st']:
            return (1, 6)  # January to June (First half)
        elif half in ['second', 'sec', '2nd']:
            return (7, 12)  # July to December (Second half)
    
    # Handle queries like "first half of {year}" or "second half of {year}"
    year_pattern = r'(\d{4})\s*(first|second|sec)\s+(half|half\s+year|half\s+yearly)'
    year_match = re.search(year_pattern, text, re.IGNORECASE)
    
    if year_match:
        year = int(year_match.group(1))  # Extract the year
        half = year_match.group(2).lower()

        if half in ['first', '1st']:
            return (1, 6)  # January to June (First half)
        elif half in ['second', 'sec', '2nd']:
            return (7, 12)  # July to December (Second half)
    if 'current' in text.lower() and 'second half' in text.lower():
        current_date = datetime.now()
        current_year = current_date.year
        return (current_year,7, 12)
    
    # Default case, use current year if no specific year is mentioned
    if 'current' in text.lower():
        current_date = datetime.now()
        current_year = current_date.year
        return (current_year, 1, 6)  # Default to first half of the current year

    return None


def extract_fortnight(text):
    pattern = r'\b(fortnight|two\s+weeks|last\s+fortnight|last\s+two\s+weeks|last\s+14\s+days)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        today = datetime.now()
        start_date = today - timedelta(days=14)
        return start_date, today
    return None


def extract_last_n_months(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_months_str = match.group(2).lower()
        num_months = int(num_months_str) if num_months_str.isdigit() else word_to_num.get(num_months_str, 0)
        
        # Get the current date
        today = datetime.now()

        # Calculate the start date using relativedelta, which considers the actual number of days in the months
        start_date = today - relativedelta(months=num_months)
        
        return start_date, today
    return None
    
# def extract_sales_for_specific_date(df, specific_date):
#     specific_date = pd.to_datetime(specific_date, errors='coerce').date()
#     if pd.isna(specific_date):
#         return None, "Error: The provided date is invalid."

#     df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce').dt.date
#     daily_sales_count = df[df['PurchaseDate'] == specific_date]['SellingPrice'].count()
#     daily_sales_price = df[df['PurchaseDate'] == specific_date]['SellingPrice'].sum()
    
#     return daily_sales_count, daily_sales_price
def extract_sales_for_specific_date(df, specific_date):
    try:
        # Convert the specific_date to a datetime object
        specific_date = pd.to_datetime(specific_date, errors='coerce').date()
        
        if pd.isna(specific_date):
            return None, "Error: The provided date is invalid."
        
        # Ensure 'PurchaseDate' is converted to datetime and handle any NaT values
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        
        # Check if the conversion was successful
        if df['PurchaseDate'].isnull().any():
            return None, "Error: Some dates in 'PurchaseDate' could not be converted."
        
        # Filter and calculate sales
        daily_sales_count = df[df['PurchaseDate'].dt.date == specific_date]['SellingPrice'].count()
        daily_sales_price = df[df['PurchaseDate'].dt.date == specific_date]['SellingPrice'].sum()
        
        return daily_sales_count, daily_sales_price
    
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def convert_to_datetime(self, date_str: str) -> datetime:
    """Converts a date string to a datetime object."""
        
    # Normalize the string for parsing
    date_str = date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        
    try:
        return datetime.strptime(date_str.strip(), "%d %B %Y")
    except ValueError:
        # Handle month-only strings (e.g., "August 2022")
        return datetime.strptime(date_str.strip(), "%B %Y")

def calculate_country_sales(df, country):
    # Filter the data for the given country
    country_data = df[df['countryname'].str.lower() == country.lower()]
    
    # Calculate the sales count and total revenue
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    
    return sales_count, total_revenue
    
def calculate_country_sales_by_year(df, country, year):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_country_sales_by_month_year(df, country, month, year):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & 
                      (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue

def extract_country_from_text(text, country_names):
    text_lower = text.lower()    
    matched_countries = [country for country in country_names if country.lower() in text_lower]
    return matched_countries


def extract_country(text: str, valid_countries: List[str]) -> List[str]:
    # Compile a regex pattern to match valid country names
    country_pattern = r'\b(?:' + '|'.join(map(re.escape, valid_countries)) + r')\b'
    
    # Find all matches for countries in the user message
    found_countries = re.findall(country_pattern, text, re.IGNORECASE)
    unique_countries = list(dict.fromkeys(found_countries))  # Preserves order and removes duplicates
    return unique_countries[:2]
    

def calculate_region_sales(df, region):
    """Calculates total sales and revenue for the given region."""
    region_sales = df[df['regionname'].str.lower() == region.lower()]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_region_sales_by_month_year(df, region, month, year):
    """Calculates total sales and revenue for a region for a specific month and year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
                      
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_region_sales_by_year(df, region, year):
    """Calculates total sales and revenue for a region for a specific year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.year == year)]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_total_region_sales(df, region):
    """Calculates total sales and revenue for the given region (all years and months)."""
    return calculate_region_sales(df, region)


def calculate_planname_sales_by_month_year(df, planname, month, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['month'] == month) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_planname_sales_by_year(df, planname, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue = filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_total_planname_sales(df, planname):
    filtered_df = df[df['PlanName'] == planname]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue


def calculate_top_sales_plans(df, year=None, month=None):
    # Implement logic to calculate top selling plans
    # Example:
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nlargest(10, 'SalesCount')

def calculate_least_sales_plans(df, year=None, month=None):
    # Implement logic to calculate least selling plans
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nsmallest(10, 'SalesCount')

def extract_sales_in_date_range(df, start_date, end_date):
    try:
        # Ensure dates are in datetime format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # Filter the DataFrame for the given date range
        filtered_df = df[(df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]

        # Calculate count and sum
        sales_count = filtered_df['SellingPrice'].count()
        total_price = filtered_df['SellingPrice'].sum()
        
        return sales_count, total_price
    except Exception as e:
        logging.error(f"Error in extract_sales_in_date_range: {e}")
        return None, f"An error occurred while processing sales data: {e}"
    


#######################TOTALSALES#############################################################################################

class ActionGetTotalSales(Action):
    def name(self) -> Text:
        return "action_get_total_sales"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            global df
            logging.info("Running ActionGetTotalSales...")
            user_message = tracker.latest_message.get('text')
            if df.empty or df['SellingPrice'].isnull().all():
                dispatcher.utter_message(text="Error: The sales data is empty or invalid.")
                return []

            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            #date_range = extract_date_range(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            total_sales_count = 0
            total_sales_price = 0.0
            specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            
           

            if specific_date_text:
                logging.info(f"Processing sales data for specific date: {specific_date_text}")
                
                daily_sales_count, daily_sales_price = extract_sales_for_specific_date(df, specific_date_text)
                if daily_sales_count is not None:
                    if daily_sales_count > 0:
                        dispatcher.utter_message(text=f"The total sales for {specific_date_text} is {daily_sales_count} with a sales price of ${daily_sales_price:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded on {specific_date_text}.")
                else:
                    dispatcher.utter_message(text=daily_sales_price)  # Return the error message from the function
                return []
            # if date_range :
            #     logging.info(f"date range...{date_range}")
            #     start_date, end_date, error_message = extract_date_range(date_range)
    
            #     if error_message:
            #         dispatcher.utter_message(text=error_message)
            #         return []
            #     logging.info(f"Extracted start_date: {start_date}, end_date: {end_date}")
            #     count_range = df[
            #         (df['PurchaseDate'] >= pd.Timestamp(start_date)) & 
            #         (df['PurchaseDate'] <= pd.Timestamp(end_date))
            #     ]['SellingPrice'].count()
                
            #     price_range = df[
            #         (df['PurchaseDate'] >= pd.Timestamp(start_date)) & 
            #         (df['PurchaseDate'] <= pd.Timestamp(end_date))
            #     ]['SellingPrice'].sum()
            #     if count_range > 0:
            #         dispatcher.utter_message(text=f"The total sales from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} is {count_range} with a sales price of ${price_range:,.2f}.")
            #     else:
            #         dispatcher.utter_message(text=f"No sales were recorded from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            # else:
            #     dispatcher.utter_message(text="No valid date range was provided.")
            # return []
                
            # Today's sales
            if today:
                today_date = datetime.now().date()
                logging.info(f"Processing today's sales...{today_date}")
                today_sales_count = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].count()
                today_sales_price = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].sum()
                dispatcher.utter_message(
                    text=f"The total sales for today ({today_date}) is {today_sales_count} with a sales price of ${today_sales_price:.2f}."
                )
                return []

            # Yesterday's sales
            if last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Processing yesterday's sales...{lastday}")
                yesterday_sales_count = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].count()
                yesterday_sales_price = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].sum()
                if yesterday_sales_count > 0:
                    dispatcher.utter_message(
                        text=f"The total sales for yesterday ({lastday}) is {yesterday_sales_count} with a sales price of ${yesterday_sales_price:.2f}."
                    )
                else:
                    dispatcher.utter_message(
                        text=f"No sales were recorded yesterday ({lastday})."
                    )
                return []
            # Handle half-year request
            if half_year:
                logging.info(f"Processing sales data for half-year... {half_year}")
                start_month, end_month = half_year
                current_year = pd.to_datetime('today').year
                half_year_sales_count = df[
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == current_year)
                ]['SellingPrice'].count()

                half_year_sales_price = df[
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == current_year)
                ]['SellingPrice'].sum()

                half_name = "First Half" if start_month == 1 else "Second Half"
                dispatcher.utter_message(
                    text=f"The total sales count for {half_name} of {current_year} is {half_year_sales_count} and sales price is ${half_year_sales_price:.2f}."
                )
                return []

            #Handle fortnight request
            if fortnight:
                logging.info(f"Processing sales data for fortnight...{fortnight}")
                start_date, end_date = fortnight
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                logging.info(f"Start date: {start_date}, End date: {end_date}")
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                count_fortnight = df[
                    (df['PurchaseDate'] >= start_date) & 
                    (df['PurchaseDate'] <=  end_date)
                ]['SellingPrice'].count()
            
                price_fortnight = df[
                    (df['PurchaseDate'] >= start_date) & 
                    (df['PurchaseDate'] <= end_date)
                ]['SellingPrice'].sum()
                if count_fortnight is not None:
                    if count_fortnight > 0:
                        dispatcher.utter_message(text=f"The total sales for the last fortnight ({start_date_formatted} to {end_date_formatted}) is {count_fortnight} with a sales price of ${price_fortnight:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded for the fortnight ({start_date_formatted} to {end_date_formatted}).")
                else:
                    dispatcher.utter_message(text=price_fortnight)  # Return the error message from the function
                return []

            #Handle last N months request
            if last_n_months:
                logging.info(f"Processing sales data for the last N months...{last_n_months}")
                start_date, end_date = last_n_months
                # Extract the number of months for display
                num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

                count_last_n_months, price_last_n_months = extract_sales_in_date_range(df, start_date, end_date)
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                if count_last_n_months is not None:
                    if count_last_n_months > 0:
                        dispatcher.utter_message(text=f"The total sales in the last  {num_months} months ({start_date_formatted} to {end_date_formatted}) is {count_last_n_months} with a sales price of ${price_last_n_months:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded in the last N months ({start_date_formatted} to {end_date_formatted}).")
                else:
                    dispatcher.utter_message(text=price_last_n_months)  # Return the error message from the function
                return []
                
            if quarterly:
                logging.info(f"Processing quarterly sales... {quarterly}")
                try:
                    start_month, end_month = quarterly
                    current_year = pd.to_datetime('today').year
                    
                    # Filter data for the quarter
                    quarterly_sales_count = df[
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) &
                        (df['PurchaseDate'].dt.year == current_year)
                    ]['SellingPrice'].count()
                    
                    quarterly_sales_price = df[
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) &
                        (df['PurchaseDate'].dt.year == current_year)
                    ]['SellingPrice'].sum()

                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                    
                    dispatcher.utter_message(
                        text=f"The total sales count for the {quarter_name} of {current_year} is {quarterly_sales_count} and sales price is ${quarterly_sales_price:.2f}."
                    )
                except Exception as e:
                    dispatcher.utter_message(
                        text=f"An error occurred while processing quarterly sales: {str(e)}"
                    )
                return []

           
  
            if month_year_pairs:
                logging.info(f"sales with month - year... {month_year_pairs}")
                try:
                    total_sales_count = total_sales_price = 0.0
                    for month, year in month_year_pairs:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price

                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales by year: {str(e)}")
                return []

            if years:
                logging.info(f"sales with year...: {years}")
                try:
                    total_sales_count = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].count()
                    total_sales_price = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].sum()
                    
                    if len(years) == 1:
                        dispatcher.utter_message(text=f"The total sales count for {years[0]} is {total_sales_count} and sales price is ${total_sales_price:.2f}.")
                    else:
                        years_str = ', '.join(map(str, years))
                        dispatcher.utter_message(text=f"Total sales count for {years_str} is {total_sales_count} and total sales price is ${total_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing annual sales: {str(e)}")
                return []

            if months_extracted:
                logging.info(f"month with current year...:  {months_extracted}")
                try:
                    current_year = pd.to_datetime('today').year
                    for month in months_extracted:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price
                        
                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {current_year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales: {str(e)}")
                return []
            logging.info("processing total sales.....")
    
            total_sales_count, total_sales_price = self.calculate_total_sales(df)
            dispatcher.utter_message(text=f"The overall total sales count is {total_sales_count} with a total sales price of ${total_sales_price:.2f}.")

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred in processing your request: {str(e)}")
        
        return []

    def calculate_total_sales(self, df: pd.DataFrame) -> Tuple[int, float]:
        total_sales_count = df['SellingPrice'].count() if 'SellingPrice' in df.columns else 0
        total_sales_price = df['SellingPrice'].sum() if 'SellingPrice' in df.columns else 0.0
        return total_sales_count, total_sales_price

###############################################COMPARESALES################################################################################

class ActionCompareSales(Action):
    def name(self) -> str:
        return "action_compare_sales"

    def run(self, dispatcher, tracker, domain) -> list[Dict[Text, Any]]:
        global df
        try:
            logging.info("Running ActionCompareSales...")
            user_message = tracker.latest_message.get('text')
            if not user_message:
                dispatcher.utter_message(text="I didn't receive any message for comparison. Please specify a time range.")
                return []
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns ('PurchaseDate', 'SellingPrice') are missing in the dataset.")
                return []

            month_pattern = r"(\w+ \d{4}) and (\w+ \d{4})"
            year_pattern = r"(\d{4}) and (\d{4})"

            # Check for month comparison
            month_matches = re.findall(month_pattern, user_message)
            if month_matches:
                logging.info("month year comparison...")
                try:
                    month1, month2 = month_matches[0]
                    start_date_1 = datetime.strptime(month1, "%B %Y").replace(day=1)
                    start_date_2 = datetime.strptime(month2, "%B %Y").replace(day=1)

                    logging.info(f"Comparing sales for {month1} and {month2}...")
                    
                    sales_count_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].count()
                    sales_price_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].count()
                    sales_price_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].sum()

                    comparison_text = (
                         f"Sales Comparison between {month1} and {month2}:\n\n"
                         f"• {month1}:\n"
                         f"   - Total Sales: {sales_count_1} sales\n"
                         f"   - Sales Revenue: ${sales_price_1:.2f}\n\n"
                         f"• {month2}:\n"
                         f"   - Total Sales: {sales_count_2} sales\n"
                         f"   - Sales Revenue: ${sales_price_2:.2f}\n\n"
                    )

                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2

                    if count_difference > 0:
                        comparison_text += f"Difference in sales: {month1} had {abs(count_difference)} more sales than {month2}.\n"
                    elif count_difference < 0:
                        comparison_text += f"Difference in sales: {month2} had {abs(count_difference)} more sales than {month1}.\n"
                    else:
                        comparison_text += " Both months had the same number of sales.\n"

                    if price_difference > 0:
                        comparison_text += f"Difference in revenue: {month1} generated ${abs(price_difference):.2f} more in sales revenue than {month2}."
                    elif price_difference < 0:
                        comparison_text += f"Difference in revenue: {month2} generated ${abs(price_difference):.2f} more in sales revenue than {month1}."
                    else:
                        comparison_text += " Both months generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_text)
                except ValueError as ve:
                    dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year'.")
                    logging.error(f"Date parsing error for month comparison: {ve}")
                return []

            # Check for year comparison
            year_matches = re.findall(year_pattern, user_message)
            if year_matches:
                logging.info("year comparison...")
                try:
                    year1, year2 = year_matches[0]
                    sales_count_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].count()
                    sales_price_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].count()
                    sales_price_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].sum()

                    comparison_text = (
                        f"Sales Comparison between {year1} and {year2}:\n\n"
                        f"• {year1}:\n"
                        f"   - Total Sales: {sales_count_1} sales\n"
                        f"   - Total Revenue: ${sales_price_1:.2f}\n\n"
                        f"• {year2}:\n"
                        f"   - Total Sales: {sales_count_2} sales\n"
                        f"   - Total Revenue: ${sales_price_2:.2f}\n\n"
                    )
                    
                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2

                    if count_difference > 0:
                        comparison_text += f"Sales Count Difference: {year1} had {abs(count_difference)} more sales than {year2}.\n"
                    elif count_difference < 0:
                        comparison_text += f"Sales Count Difference: {year2} had {abs(count_difference)} more sales than {year1}.\n"
                    else:
                        comparison_text += " Both years had the same number of sales."

                    if price_difference > 0:
                        comparison_text += f"Revenue Difference: {year1} generated ${abs(price_difference):.2f} more in sales revenue than {year2}."
                    elif price_difference < 0:
                        comparison_text += f"Revenue Difference: {year2} generated ${abs(price_difference):.2f} more in sales revenue than {year1}."
                    else:
                        comparison_text += " Both years generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_text)
                except ValueError as ve:
                    dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year'.")
                    logging.error(f"Date parsing error for year comparison: {ve}")
                return []

            dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year'.")
        
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            logging.error(f"Unexpected error in ActionCompareSales: {e}")
        
        return []




##############################################################salesforcountry###########################################################################


# class ActionCountrySales(Action):

#     def name(self) -> str:
#         return "action_country_sales"
   
#     def run(self, dispatcher: CollectingDispatcher, tracker, domain):
#         global df
#         logging.info("Running ActionCountrySales...")

#         try:
#             # Fetch sales data from the database dynamically
           

#             # Check if the dataframe is empty
#             if df.empty:
#                 dispatcher.utter_message(text="Sales data could not be retrieved from the database. Please try again later.")
#                 logging.error("Sales data is empty after fetching from the database.")
#                 return []

#             user_message = tracker.latest_message.get('text')
#             logging.info(f"User message: {user_message}")

#             country_names = df['countryname'].dropna().unique().tolist()

#             # Extract country name from the user message or slot
#             country = next(tracker.get_latest_entity_values('country'), None)
#             logging.info(f"Extracted country: {country}")

#             # If country is not detected, ask the user to provide a valid country
#             if not country:
#                 dispatcher.utter_message(text="Please provide a valid country name.")
#                 logging.info("Country not provided by the user.")
#                 return []

#             # Check if the country exists in the dataset
#             if country not in df['countryname'].values:
#                 dispatcher.utter_message(text=f"Sorry, we do not have sales data for {country}. Please provide another country.")
#                 logging.info(f"Country {country} not found in the dataset.")
#                 return []

#             # Extract years, months, and month-year pairs from the user message
#             try:
#                 years = extract_years(user_message)
#                 months_extracted = extract_months(user_message)
#                 month_year_pairs = extract_month_year(user_message)
#             except Exception as e:
#                 dispatcher.utter_message(text="There was an error extracting dates from your message. Please try specifying the month and year clearly.")
#                 logging.error(f"Date extraction error: {e}")
#                 return []

#             logging.info(f"Processing sales data for country: {country}")
#             response_message = ""

#             try:
#                 # Use helper functions to calculate sales count and revenue for the country
#                 if month_year_pairs:
#                     logging.info("Processing request for specific month and year sales data...")
#                     for month, year in month_year_pairs:
#                         sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, year)
#                         response_message += f"In {months_reverse[month].capitalize()} {year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
#                 elif years:
#                     logging.info("Processing request for specific year sales data...")
#                     for year in years:
#                         sales_count, total_revenue = calculate_country_sales_by_year(df, country, year)
#                         response_message += f"In {year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
#                 elif months_extracted:
#                     current_year = datetime.now().year
#                     logging.info("Processing request for specific month in the current year...")
#                     for month in months_extracted:
#                         sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, current_year)
#                         response_message += f"In {months_reverse[month].capitalize()} {current_year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
#                 else:
#                     # If no specific month or year, return total sales for the country
#                     sales_count, total_revenue = calculate_country_sales(df, country)
#                     response_message = f"In {country}, there have been a total of {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
#             except Exception as e:
#                 dispatcher.utter_message(text="An error occurred while calculating sales data. Please try again.")
#                 logging.error(f"Sales calculation error for country {country}: {e}")
#                 return []

#             dispatcher.utter_message(text=response_message)
#             return []
        
#         except Exception as e:
#             dispatcher.utter_message(text="An error occurred while processing your request. Please try again later.")
#             logging.error(f"Error fetching or processing sales data: {e}")
#             return []




class ActionCountrySales(Action):
    def name(self) -> str:
        return "action_country_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        global df
        logging.info("Running ActionCountrySales...")

        try:
            # Check if the dataframe is empty
            if df.empty:
                dispatcher.utter_message(text="Sales data could not be retrieved from the database. Please try again later.")
                logging.error("Sales data is empty after fetching from the database.")
                return []

            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")

            # Get the list of country names from the dataset
            country_names = df['countryname'].dropna().unique().tolist()

            # Extract country from the user message
            country_extracted = next(tracker.get_latest_entity_values('country'), None)
            logging.info(f"Initial extracted country: {country_extracted}")

            if not country_extracted:
                # Attempt fuzzy matching for country extraction if not directly found
                matched_country = process.extractOne(user_message, country_names, scorer=lambda x, y: x.lower() in y.lower())
                if matched_country and matched_country[1] > 75:  # Threshold for matching confidence
                    country_extracted = matched_country[0]
                else:
                    dispatcher.utter_message(text="Please provide a valid country name.")
                    logging.info("Country not provided or could not be matched.")
                    return []

            # Standardize case-insensitive matching
            country = next((c for c in country_names if c.lower() == country_extracted.lower()), None)

            if not country:
                dispatcher.utter_message(text=f"Sorry, we do not have sales data for {country_extracted}. Please provide another country.")
                logging.info(f"Country {country_extracted} not found in the dataset.")
                return []

            # Extract years, months, and month-year pairs from the user message
            try:
                years = extract_years(user_message)
                months_extracted = extract_months(user_message)
                month_year_pairs = extract_month_year(user_message)
                quarterly = extract_quarters_from_text(user_message)
                half_year = extract_half_year_from_text(user_message)
                fortnight = extract_fortnight(user_message)
                last_n_months = extract_last_n_months(user_message)
                date_range = extract_date_range(user_message)
                today = extract_today(user_message)
                last_day = extract_last_day(user_message)
                specific_date_text = next(tracker.get_latest_entity_values("specific_date"), None)
            except Exception as e:
                dispatcher.utter_message(text="There was an error extracting dates from your message. Please try specifying the month and year clearly.")
                logging.error(f"Date extraction error: {e}")
                return []

            logging.info(f"Processing sales data for country: {country}")
            response_message = ""

            try:
                # Use helper functions to calculate sales count and revenue for the country
                if month_year_pairs:
                    logging.info("Processing request for specific month and year sales data...")
                    for month, year in month_year_pairs:
                        sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, year)
                        response_message += f"In {months_reverse[month].capitalize()} {year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                elif years:
                    logging.info("Processing request for specific year sales data...")
                    for year in years:
                        sales_count, total_revenue = calculate_country_sales_by_year(df, country, year)
                        response_message += f"In {year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                elif months_extracted:
                    current_year = datetime.now().year
                    logging.info("Processing request for specific month in the current year...")
                    for month in months_extracted:
                        sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, current_year)
                        response_message += f"In {months_reverse[month].capitalize()} {current_year}, {country} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                else:
                    # If no specific month or year, return total sales for the country
                    sales_count, total_revenue = calculate_country_sales(df, country)
                    response_message = f"In {country}, there have been a total of {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
            except Exception as e:
                dispatcher.utter_message(text="An error occurred while calculating sales data. Please try again.")
                logging.error(f"Sales calculation error for country {country}: {e}")
                return []

            dispatcher.utter_message(text=response_message)
            return []

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again later.")
            logging.error(f"Error fetching or processing sales data: {e}")
            return []


##########################################################################################################################PLANRELATEDSALES#################

class ActionPlanNameByCountry(Action):
    def name(self) -> str:
        return "action_planname_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionPlanNameByCountry...")
        global df

        try:
        #     

            # Get user message and extract entities
            user_message = tracker.latest_message.get('text')
            
            country = next(tracker.get_latest_entity_values('countryname'), None)
            planname = next(tracker.get_latest_entity_values('planname'), None)

            # Check for a valid country input
            if not country:
                dispatcher.utter_message(text="Please provide a valid country name.")
                logging.warning("Country name not provided by the user.")
                return []
            elif country not in df['countryname'].values:
                dispatcher.utter_message(text=f"Sorry, we do not have data for {country}. Please try another country.")
                logging.warning(f"Country {country} not found in the dataset.")
                return []

            # Extract plans for the specified country
            country_plans = df[df['countryname'] == country]['PlanName'].unique()
            if len(country_plans) == 0:
                dispatcher.utter_message(text=f"No plans available for {country}.")
                logging.info(f"No plans found for country: {country}")
                return []

            # Extract years, months, and month-year pairs from user message
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            logging.info(f"Processing sales data for country: {country} and plans: {planname}")
            response_message = ""

            # Helper function to format sales and revenue
            def format_sales_data(planname, sales_count, total_revenue):
                return (f"Plan: {planname}\n"
                        f"Total Sales: {sales_count}\n"
                        f"Revenue: ${total_revenue:,.2f}")

            # Generate response based on provided filters
            if month_year_pairs:
                for month, year in month_year_pairs:
                    response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, year)
                        if sales_count > 0 and total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📅 Sales Overview for {year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_year(df, plan, year)
                        if sales_count > 0 or total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, current_year)
                        if sales_count > 0 or total_revenue > 0:
                            response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            else:
                response_message += f"📊 Total Sales Overview for {country} Plans:\n\n"
                for plan in country_plans:
                    sales_count, total_revenue = calculate_total_planname_sales(df, plan)
                    if sales_count > 0 or total_revenue > 0:
                        response_message += format_sales_data(plan, sales_count, total_revenue) + "\n\n"

            # Check if no data was found
            if not response_message.strip():
                dispatcher.utter_message(text=f"No sales data found for the specified criteria in {country}.")
                logging.info("No sales data found for specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except KeyError as e:
            logging.error(f"KeyError: {e}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again.")
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            dispatcher.utter_message(text="There was an issue with the input data format. Please specify the dates correctly.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
        
        return []


#################################################################################################################active plans and country name##############


class ActionGetActiveAndInactivePlans(Action):
    def name(self) -> str:
        return "action_get_active_inactive_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActivePlans...")
        global df

        try:
        

            user_message = tracker.latest_message.get('text')
            
            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive plans
            active_plans_current_year = df[df['PurchaseDate'].dt.year == current_year]['PlanName'].unique()
            past_plans = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['PlanName'].unique()
            active_plans = list(active_plans_current_year)
            inactive_plans = list(set(past_plans) - set(active_plans))
            plans_count = len(active_plans)
            inactive_plans_count = len(inactive_plans)

            # Generate response
            if plans_count == 0 and inactive_plans_count == 0:
                dispatcher.utter_message(text="No active or inactive plans found for the specified criteria.")
                logging.info("No plans found for the specified criteria.")
                return []

            response = f"Total Active Plans: {plans_count}\n\n"
            response += "HERE ARE THE ACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in active_plans) + "\n\n"
            response += f"Total Inactive Plans: {inactive_plans_count}\n\n"
            response += "HERE ARE THE INACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in inactive_plans)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactivePlans: {e}")
        
        return []


class ActionGetActiveAndInactiveCountries(Action):
    def name(self) -> str:
        return "action_get_active_inactive_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActiveCountry...")
        global df

        try:
        #     
            user_message = tracker.latest_message.get('text')
            

            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive countries
            active_countries_current_year = df[df['PurchaseDate'].dt.year == current_year]['countryname'].unique()
            past_countries = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['countryname'].unique()
            active_countries = list(active_countries_current_year)
            inactive_countries = list(set(past_countries) - set(active_countries))
            countries_count = len(active_countries)
            inactive_countries_count = len(inactive_countries)

            # Generate response
            if countries_count == 0 and inactive_countries_count == 0:
                dispatcher.utter_message(text="No active or inactive countries found for the specified criteria.")
                logging.info("No countries found for the specified criteria.")
                return []

            response = f"Total Active Countries: {countries_count}\n\n"
            response += "HERE ARE THE ACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in active_countries) + "\n\n"
            response += f"Total Inactive Countries: {inactive_countries_count}\n\n"
            response += "HERE ARE THE INACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in inactive_countries)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactiveCountries: {e}")
        
        return []



#############################################################################################################top and lowest sales plan############


class ActionTopPlansSales(Action):

    def name(self) -> str:
        return "action_top_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionTopPlansSales...")
        
        global df

        

        user_message = tracker.latest_message.get('text')

        # Extract year and month from user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        logging.info(f"Processing top plans sales data based on user request: {user_message}")
        response_message = ""

        # Helper function to format the output
        def format_sales_data(planname, sales_count, total_revenue):
            return (f"Plan: {planname}\n "
                    f"Total Sales: {sales_count}\n "  
                    f"Revenue: ${total_revenue:,.2f}")

        try:
            # Determine the response based on user input
            if month_year_pairs:
                month, year = month_year_pairs[0]
                response_message += f"📈 Top 10 Plans in {months_reverse[month].capitalize()} {year}:\n\n"
                top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += "No data available for the specified month and year.\n"
                else:
                    for _, row in top_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📈 Top 10 Plans in {year}:\n\n"
                    top_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No data available for the year {year}.\n"
                    else:
                        for _, row in top_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📈 Top 10 Plans in {months_reverse[month].capitalize()} {current_year}:\n\n"
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No data available for the month {months_reverse[month]} {current_year}.\n"
                    else:
                        for _, row in top_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            else:
                response_message += "📈 Top 10 Highest Sales Plans Overall:\n\n"
                top_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += "No data available overall.\n"
                else:
                    for _, row in top_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

        except Exception as e:
            logging.error(f"Error while processing top sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []

class ActionLowestPlansSales(Action):

    def name(self) -> str:
        return "action_lowest_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionLowestPlansSales...")

        global df



        user_message = tracker.latest_message.get('text')
        

        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        response_message = ""

        # Helper function to format sales data
        def format_sales_data(planname, sales_count, total_revenue):
            return (f"Plan: {planname} \n "
                    f"Total Sales: {sales_count} \n "
                    f"Revenue: ${total_revenue:,.2f}")

        try:
            if month_year_pairs:
                month, year = month_year_pairs[0]
                response_message += f"📅 Lowest Sales in {months_reverse[month].capitalize()} {year}:\n\n"
                lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += "No data available for the specified month and year.\n"
                else:
                    for _, row in lowest_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif years:
                for year in years:
                    response_message += f"📅 Lowest Sales in {year}:\n\n"
                    lowest_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No data available for the year {year}.\n"
                    else:
                        for _, row in lowest_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            elif months_extracted:
                current_year = datetime.now().year
                for month in months_extracted:
                    response_message += f"📅 Lowest Sales in {months_reverse[month].capitalize()} {current_year}:\n\n"
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No data available for {months_reverse[month]} {current_year}.\n"
                    else:
                        for _, row in lowest_plans.iterrows():
                            response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

            else:
                response_message += "📊 Overall Lowest Sales Plans:\n\n"
                lowest_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += "No data available overall.\n"
                else:
                    for _, row in lowest_plans.iterrows():
                        response_message += format_sales_data(row['PlanName'], row['total_sales'], row['total_revenue']) + "\n\n"

        except Exception as e:
            logging.error(f"Error while processing lowest sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []



        
########################################################################################################### top highest and lowest country sales #########################
class ActionTopHighestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_highest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("f top highest sales by country")
        global df

        
        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = "📊 Top 10 Highest Sales Countries:\n\n"

        # Helper function to calculate top 10 highest sales by country
        def calculate_top_highest_sales_by_country(df, year=None, month=None):
            df_filtered = df.copy()

            try:
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by country and calculate sales count and total revenue
                df_grouped = df_filtered.groupby('countryname').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).sort_values('TotalRevenue', ascending=False).head(10).reset_index()

                return df_grouped
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing sales data: {str(e)}")
                return pd.DataFrame()

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            for month, year in month_year_pairs:
                top_sales = calculate_top_highest_sales_by_country(df, year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only year is provided, show results for the entire year
        elif years:
            for year in years:
                top_sales = calculate_top_highest_sales_by_country(df, year)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales = calculate_top_highest_sales_by_country(df, current_year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {current_year}:\n"
                for index, row in top_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If no filters, show overall top 10 highest sales by country
        else:
            top_sales = calculate_top_highest_sales_by_country(df)
            if top_sales.empty:
                dispatcher.utter_message(text="No sales data found.")
                return []
            for index, row in top_sales.iterrows():
                response_message += (
                    f"- {row['countryname']}: \n"
                    f"  Sales Count: {row['SalesCount']} \n"
                    f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
        return []


class ActionTopLowestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_lowest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info(f"top lowest sales by country")
        global df

        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = "📉 Top 10 Lowest Sales Countries:\n\n"

        def calculate_top_lowest_sales_by_country(df: pd.DataFrame, year: int = None, month: str = None) -> pd.DataFrame:
            df_filtered = df.copy()

            try:
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by country and calculate sales metrics
                df_grouped = df_filtered.groupby('countryname').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).sort_values('TotalRevenue').head(10).reset_index()
                return df_grouped
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing sales data: {str(e)}")
                return pd.DataFrame()

        if month_year_pairs:
            for month, year in month_year_pairs:
                lowest_sales = calculate_top_lowest_sales_by_country(df, year, month)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        elif years:
            for year in years:
                lowest_sales = calculate_top_lowest_sales_by_country(df, year)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year}.")
                    continue
                response_message += f"🔍 Sales Overview for {year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                lowest_sales = calculate_top_lowest_sales_by_country(df, current_year, month)
                if lowest_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year}.")
                    continue
                response_message += f"🔍 Sales Overview for {month} {current_year}:\n"
                for index, row in lowest_sales.iterrows():
                    response_message += (
                        f"- {row['countryname']}: \n"
                        f"  Sales Count: {row['SalesCount']} \n"
                        f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        else:
            lowest_sales = calculate_top_lowest_sales_by_country(df)
            if lowest_sales.empty:
                dispatcher.utter_message(text="No sales data found.")
                return []
            for index, row in lowest_sales.iterrows():
                response_message += (
                    f"- {row['countryname']}: \n"
                    f"  Sales Count: {row['SalesCount']} \n"
                    f"  Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        dispatcher.utter_message(text=response_message)
        return []




#################################################compare countries sales#######################################################################

class ActionCompareCountries(Action):
    def name(self) -> Text:
        return "action_compare_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info(f"Comparing country sales")
        global df

        
        user_message = tracker.latest_message.get('text')
    

        logging.info(f"User message: {user_message}")

        # Extract countries, year, and month
        countries = extract_country(user_message, df['countryname'].dropna().unique().tolist())
        year = extract_years(user_message)  # Implement logic to extract year from message
        month = extract_months(user_message)  # Implement logic to extract month from message
        logging.info(f"Extracted countries: {countries}, year: {year}, month: {month}")

        # Validate that two countries are provided for comparison
        if len(countries) != 2:
            detected_message = f"Detected countries: {', '.join(countries)}" if countries else "No countries detected"
            dispatcher.utter_message(text="Please provide two countries for comparison.")
            return []

        country1, country2 = countries[0], countries[1]

        # Filter data by countries and time period
        try:
            if month and year:
                # Compare specific month and year
                df_country1 = filter_by_country_year_month(df, country=country1, year=year, month=month)
                df_country2 = filter_by_country_year_month(df, country=country2, year=year, month=month)
                comparison_type = f"{month}/{year}"
            elif year:
                # Compare whole year
                df_country1 = filter_by_country_year(df, country=country1, year=year)
                df_country2 = filter_by_country_year(df, country=country2, year=year)
                comparison_type = f"Year {year}"
            else:
                dispatcher.utter_message(text="Please specify a valid year or month and year for comparison.")
                return []

            # Check if filtered data is empty
            if df_country1.empty or df_country2.empty:
                dispatcher.utter_message(text=f"No sales data found for {country1} or {country2} for {comparison_type}.")
                return []

            # Generate comparison results
            comparison_result = self.compare_sales(df_country1, df_country2, country1, country2, comparison_type)
            dispatcher.utter_message(text=f"Comparison of sales between {country1} and {country2} for {comparison_type}:\n\n{comparison_result}")
        
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            dispatcher.utter_message(text="An error occurred while comparing sales. Please try again later.")
            return []

        return []

    def compare_sales(self, df_country1, df_country2, country1, country2, comparison_type):
        result = f"Sales Comparison for {comparison_type}:\n\n"

        try:
            # Total sales comparison
            total_sales_amount_country1 = df_country1['SellingPrice'].sum()
            total_sales_amount_country2 = df_country2['SellingPrice'].sum()
            total_sales_count_country1 = df_country1['SellingPrice'].count()
            total_sales_count_country2 = df_country2['SellingPrice'].count()

            result += f"Total Sales in {country1}: {total_sales_count_country1} sales, generating ${total_sales_amount_country1:,.2f}\n"
            result += f"Total Sales in {country2}: {total_sales_count_country2} sales, generating ${total_sales_amount_country2:,.2f}\n\n"

            # Calculate sales count and amount differences with descriptive messages
            sales_count_difference = abs(total_sales_count_country1 - total_sales_count_country2)
            sales_amount_difference = abs(total_sales_amount_country1 - total_sales_amount_country2)

            if total_sales_amount_country1 > total_sales_amount_country2:
                result += f"{country1} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country2}.\n\n"
            elif total_sales_amount_country1 < total_sales_amount_country2:
                result += f"{country2} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country1}.\n\n"
            else:
                result += f"Sales revenue or counts were equal between {country1} and {country2}.\n\n"

            # Generate top and least plans and sources
            result += f"Top 5 plans by sales in {country1}:\n"
            result += self.get_top_plans(df_country1)
            result += f"\nTop 5 plans by sales in {country2}:\n"
            result += self.get_top_plans(df_country2)

            result += f"\nLeast 5 plans by sales in {country1}:\n"
            result += self.get_least_plans(df_country1)
            result += f"\nLeast 5 plans by sales in {country2}:\n"
            result += self.get_least_plans(df_country2)

            result += f"\nTop 5 sources by sales in {country1}:\n"
            result += self.get_top_sources(df_country1)
            result += f"\nTop 5 sources by sales in {country2}:\n"
            result += self.get_top_sources(df_country2)

            result += f"\nTop 3 payment gateways by sales in {country1}:\n"
            result += self.get_top_payment_gateways(df_country1)
            result += f"\nTop 3 payment gateways by sales in {country2}:\n"
            result += self.get_top_payment_gateways(df_country2)

            result += f"\nTop 6 refsite by sales in {country1}:\n"
            result += self.get_top_refsites(df_country1)
            result += f"\nTop 6 refsite by sales in {country2}:\n"
            result += self.get_top_refsites(df_country2)

        except Exception as e:
            logging.error(f"Error in sales comparison: {e}")
            return "An error occurred while generating the sales comparison."

        return result

    def get_top_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            return "\n".join([f"{plan}: ${sales:,.2f} with {count} sales" for plan, (sales, count) in plans_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top plans: {e}")
            return "Could not retrieve top plans."

    def get_least_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nsmallest(5, 'total_sales')
            return "\n".join([f"{plan}: ${sales:,.2f} with {count} sales" for plan, (sales, count) in plans_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching least plans: {e}")
            return "Could not retrieve least plans."

    def get_top_sources(self, df):
        try:
            source_counts = df.groupby('source').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            return "\n".join([f"{source}: ${sales:,.2f} with {count} sales" for source, (sales, count) in source_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top sources: {e}")
            return "Could not retrieve top sources."

    def get_top_payment_gateways(self, df):
        try:
            gateway_counts = df.groupby('payment_gateway').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(3, 'total_sales')
            return "\n".join([f"{gateway}: ${sales:,.2f} with {count} transactions" for gateway, (sales, count) in gateway_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top payment gateways: {e}")
            return "Could not retrieve top payment gateways."

    def get_top_refsites(self, df):
        try:
            refsite_counts = df.groupby('Refsite').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(6, 'total_sales')
            return "\n".join([f"{refsite}: ${sales:,.2f} with {count} transactions" for refsite, (sales, count) in refsite_counts.iterrows()])
        except Exception as e:
            logging.error(f"Error fetching top refsites: {e}")
            return "Could not retrieve top refsites."




##############################################################################################################most and least buying sales plan for each country#########################
    
class ActionMostAndLeastSoldPlansForCountry(Action):
    def name(self) -> str:
        return "action_most_and_least_sold_plans_for_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info(f"most and leat sold plans for country")
        global df
        
        
        user_message = tracker.latest_message.get('text')

        # Extract filters from the user's message
        country = tracker.get_slot("country")
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = f"📊 Sales Overview for {country}:\n\n"

        # Helper function to calculate most and least sold plans
        def calculate_sales_by_country(df, country, year=None, month=None):
            try:
                df_filtered = df.copy()
                
                # Filter by country
                df_filtered = df_filtered[df_filtered['countryname'].str.lower() == country.lower()]
                if year:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.year == year]
                if month:
                    df_filtered = df_filtered[df_filtered['PurchaseDate'].dt.month == month]

                # Group by plan name and calculate sales count and total revenue
                df_grouped = df_filtered.groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()

                return df_grouped

            except Exception as e:
                dispatcher.utter_message(text=f"An error occurred while calculating sales data: {str(e)}")
                return pd.DataFrame()  # Return an empty DataFrame on error

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            for month, year in month_year_pairs:
                top_sales = calculate_sales_by_country(df, country, year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {month} {year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only year is provided, show results for the entire year
        elif years:
            for year in years:
                top_sales = calculate_sales_by_country(df, country, year)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales = calculate_sales_by_country(df, country, current_year, month)
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for {month} {current_year} in {country}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {month} {current_year} in {country}:\n"
                for index, row in most_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                for index, row in least_sold.iterrows():
                    response_message += (
                        f"  - {row['PlanName']}\n"
                        f"    Sales Count: {row['SalesCount']}\n"
                        f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                    )
                response_message += "\n"

        # If no filters, show overall top 10 highest sold plans by country
        else:
            top_sales = calculate_sales_by_country(df, country)
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for {country}.")
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Overall Most Sold Plans in {country}:\n"
            for index, row in most_sold.iterrows():
                response_message += (
                    f"  - {row['PlanName']}\n"
                    f"    Sales Count: {row['SalesCount']}\n"
                    f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )
            
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n 🔍 Overall Least Sold Plans in {country}:\n"
            for index, row in least_sold.iterrows():
                response_message += (
                    f"  - {row['PlanName']}\n"
                    f"    Sales Count: {row['SalesCount']}\n"
                    f"    Total Revenue: ${row['TotalRevenue']:,.2f}\n"
                )

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
        return []





#########################################################################################################################calculate refsite,source and payment gateway sales########

class ActionSalesBySourcePaymentGatewayRefsite(Action):

    def name(self) -> str:
        return "action_sales_by_source_payment_gateway_refsite"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionSalesBySourcePaymentGatewayRefsite...")
        
        global df

        

        user_message = tracker.latest_message.get('text')

        
        # Check if required columns are present
        required_columns = ['PurchaseDate', 'SellingPrice', 'source', 'payment_gateway', 'Refsite']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            dispatcher.utter_message(text=f"Data is missing the following required columns: {', '.join(missing_columns)}")
            return []

        source = next(tracker.get_latest_entity_values('source'), None)
        payment_gateway = next(tracker.get_latest_entity_values('payment_gateway'), None)
        refsite = next(tracker.get_latest_entity_values('refsite'), None)

        # Normalize source column by converting to lowercase
        df['source'] = df['source'].str.lower()

        # Extract time conditions from the user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)

        response_message = ""

        # Helper function to format sales data
        def format_sales_data(category, name, total_revenue, sales_count):
            return (f"{category.capitalize()}: {name}\n"
                    f"Total Revenue: ${total_revenue:,.2f}\n"
                    f"Sales Count: {sales_count}\n\n")

        # Generate sales data based on time conditions
        try: 
            if month_year_pairs:
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[month_year_pairs[0][0]].capitalize()} {month_year_pairs[0][1]}:\n\n"
                for month, year in month_year_pairs:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]
                    if  filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            elif years:
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {years[0]}:\n\n"
                for year in years:
                    filtered_df = df[df['PurchaseDate'].dt.year == year]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            elif months_extracted:
                current_year = datetime.now().year
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[months_extracted[0]].capitalize()} {current_year}:\n\n"
                for month in months_extracted:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {current_year}.\n" 
                    else:
                        response_message += self.process_sales_data(filtered_df, format_sales_data)

            else:
                response_message += "📊 Overall Sales Overview by Source, Payment Gateway, and Refsite:\n\n"
                response_message += self.process_sales_data(df, format_sales_data)

            # If no data is available after processing, inform the user
            if not response_message.strip():
                dispatcher.utter_message(text="No sales data found for the specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"An error occurred while processing sales data: {str(e)}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again later.")

        return []

    def process_sales_data(self, filtered_df, format_sales_data):
        """Processes sales data to get highest and lowest sales for each category."""
        response = ""

        # Process sources
        if 'source' in filtered_df.columns:
            response += "Sales Overview by Source:\n"
            response += self.get_top_n_sales(filtered_df, 'source', format_sales_data)

        # Process payment gateways
        if 'payment_gateway' in filtered_df.columns:
            response += "\nSales Overview by Payment Gateway:\n"
            response += self.get_top_n_sales(filtered_df, 'payment_gateway', format_sales_data)

        # Process refsites
        if 'Refsite' in filtered_df.columns:
            response += "\nSales Overview by Refsite:\n"
            response += self.get_top_n_sales(filtered_df, 'Refsite', format_sales_data)

        return response

    def get_top_n_sales(self, df, column, format_sales_data, n=5):
        """Helper function to calculate top N highest and lowest sales."""
        response = ""

        # Aggregate data by column
        sales_summary = df.groupby(column)['SellingPrice'].agg(['sum', 'count']).reset_index()
        sales_summary.columns = [column, 'TotalRevenue', 'SalesCount']

        # Get top N highest sales
        top_sales = sales_summary.nlargest(n, 'SalesCount')
        response += f"Top {n} Highest Sales by {column.capitalize()}:\n"
        for index, row in top_sales.iterrows():
            response += format_sales_data(column, row[column], row['TotalRevenue'], row['SalesCount'])

        # Get top N lowest sales
        lowest_sales = sales_summary.nsmallest(n, 'SalesCount')
        response += f"\nTop {n} Lowest Sales by {column.capitalize()}:\n"
        for index, row in lowest_sales.iterrows():
            response += format_sales_data(column, row[column], row['TotalRevenue'], row['SalesCount'])

        return response

################################################################################################################calculate total sales for each month and each year, sales growth ####################################

class ActionCalculateSalesMetricsAndGrowth(Action):

    def name(self) -> str:
        return "action_calculate_sales_metrics_and_growth"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculateSalesMetrics...")

        global df

        

        try:
            
            # Check if the necessary columns exist
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns 'PurchaseDate' or 'SellingPrice' are missing from the dataset.")
                return []

            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate', 'SellingPrice'])  # Drop rows with invalid dates or sales data

            if df.empty:
                dispatcher.utter_message(text="No valid sales data available for analysis.")
                return []

            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year

            # Calculate monthly sales totals and counts
            monthly_data = df.groupby("MonthYear").agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            # Prepare the response message
            response_message = "📊 Sales Metrics Overview:\n\n"

            # Sales Overview
            response_message += " 📅 Monthly Sales Summary:\n"
            if monthly_data.empty:
                response_message += "No monthly sales data available.\n\n"
            else:
                for index, row in monthly_data.iterrows():
                    response_message += (
                        f" {row['MonthYear']} \n"
                        f" Total Sales: ${row['TotalSales']:,.2f}\n"
                        f" Sales Count: {row['SalesCount']}\n\n"
                    )

            # Prepare yearly summary
            yearly_summary = df.groupby('Year').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            # Add yearly data to the response
            response_message += " 📅 Yearly Sales Summary:\n"
            if yearly_summary.empty:
                response_message += "No yearly sales data available.\n\n"
            else:
                for index, row in yearly_summary.iterrows():
                    response_message += (
                        f"{row['Year']}\n"
                        f"Total Sales: ${row['TotalSales']:,.2f} \n"
                        f"Sales Count: {row['SalesCount']}\n\n"
                    )

            # # Calculate growth percentages
            # monthly_growth = monthly_data['SalesCount'].pct_change() * 100
            # yearly_growth = yearly_summary['SalesCount'].pct_change() * 100

            # # Growth Overview
            # response_message += "Growth Overview:\n\n"
            
            # # Monthly Growth
            # response_message += " 📈 Monthly Growth Percentage:\n"
            # for index in range(1, len(monthly_data)):
            #     growth = monthly_growth[index]
            #     response_message += (
            #         f"From {monthly_data['MonthYear'][index-1]} to {monthly_data['MonthYear'][index]}  \n"
            #         f"Growth: {growth:.2f}%\n\n"
            #     )

            # # Yearly Growth
            # response_message += "📈 Yearly Growth Percentage:\n"
            # for index in range(1, len(yearly_summary)):
            #     growth = yearly_growth[index]
            #     response_message += (
            #         f"From {yearly_summary['Year'][index-1]} to {yearly_summary['Year'][index]}  \n"
            #         f"Growth: {growth:.2f}%\n\n"
            #     )

            # # Send the formatted message
            # dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"Error while calculating sales metrics: {str(e)}")
            dispatcher.utter_message(text="An error occurred while calculating sales metrics. Please try again later.")

        return []


###################################################################################################################repeated registered emails###################################################
class ActionCountRepeatedEmails(Action):
    def name(self) -> str:
        return "action_count_repeated_emails"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCountRepeatedEmails...")

        global df

        try:
            # Check if the necessary columns are present
            required_columns = ['Email', 'PlanName', 'PurchaseDate']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                dispatcher.utter_message(text="The required columns for processing are missing in the dataset.")
                return []

            email_counts = df['Email'].value_counts()
            repeated_emails = email_counts[email_counts > 1].index.tolist()

            if repeated_emails:
                # Filter rows for repeated emails efficiently
                repeated_data = df[df['Email'].isin(repeated_emails)]
                
                # Use groupby to aggregate data for repeated emails and count the repetitions
                grouped = (
                    repeated_data.groupby('Email', as_index=False)
                    .agg(
                        Count=('Email', 'size'),
                        Plans=('PlanName', list),  # Collect all plans as a list
                        PurchaseDates=('PurchaseDate', list)  # Collect all purchase dates as a list
                    )
                    .to_dict('records')
                )

                response_lines = [f"There are {len(repeated_emails)} repeated emails:"]
                total_occurrences = 0
                for entry in grouped:
                    total_occurrences += entry['Count']
                    response_lines.append(f"\nEmail: {entry['Email']}")
                    response_lines.append(f"Repeated {entry['Count']} times")
                    response_lines.append(f"Plans:")
                    for plan, date in zip(entry['Plans'], entry['PurchaseDates']):
                        response_lines.append(f"  - Plan: {plan}, Purchase Date: {date}")
                response_lines.append(f"\nTotal number of repeated emails: {len(repeated_emails)}")
                response_lines.append(f"Total occurrences of repeated emails: {total_occurrences}")


                response_text = "\n".join(response_lines)
            else:
                response_text = "There are no repeated emails in the data."

        except Exception as e:
            logging.error(f"Error processing repeated email details: {e}")
            dispatcher.utter_message(text="An error occurred while retrieving repeated email details. Please try again later.")
            return []

        dispatcher.utter_message(text=response_text)
        return []
####################################################################################################################################profit margin#######################################
class ActionGetProfitMargin(Action):
    def name(self) -> Text:
        return "action_get_profit_margin"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            global df
            logging.info("Running ActionGetProfitMargin...")

            if df.empty or df['SellingPrice'].isnull().all() or df['CompanyBuyingPrice'].isnull().all():
                dispatcher.utter_message(text="Error: The sales data is empty or invalid.")
                return []

            # Add a ProfitMargin column to the DataFrame
            df['ProfitMargin'] = df['SellingPrice'] - df['CompanyBuyingPrice']

            user_message = tracker.latest_message.get('text')
            specific_date_text = next(tracker.get_latest_entity_values("specific_date"), None)
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)

            total_profit_margin = 0.0
            today = pd.to_datetime('today').date()
            today_profit_margin = next(tracker.get_latest_entity_values("today_profit_margin"), None)
            last_day = next(tracker.get_latest_entity_values("last_day_profit_margin"), None)
            fortnight = next(tracker.get_latest_entity_values("fortnight_profit_margin"), None)
            quarterly = next(tracker.get_latest_entity_values("quarter_profit_margin"), None)
            last_months = next(tracker.get_latest_entity_values("last_n_months_profit_margin"), None)
            half_year = next(tracker.get_latest_entity_values("half_yearly_profit_margin"), None)

            # Handle specific date
            if specific_date_text:
                logging.info("profit margin of specific date")
                specific_date = pd.to_datetime(specific_date_text).date()
                daily_profit_margin = df[df['PurchaseDate'].dt.date == specific_date]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for {specific_date} is ${daily_profit_margin:.2f}."
                )
                return []

            # Handle today's profit margin
            if today_profit_margin:
                logging.info("profit margin of today")
                today_profit_margin = df[df['PurchaseDate'].dt.date == today]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for today ({today}) is ${today_profit_margin:.2f}."
                )
                return []
            if last_day:
                logging.info("profit margin of last day")
                yesterday = today - pd.Timedelta(days=1)
                yesterday_profit_margin = df[df['PurchaseDate'].dt.date == yesterday]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for yesterday ({yesterday}) is ${yesterday_profit_margin:.2f}."
                )
                return []
            if fortnight:
                logging.info("profit margin of fortnight")
                fortnight_start = today - pd.Timedelta(days=14)
                fortnight_profit_margin = df[
                    (df['PurchaseDate'].dt.date >= fortnight_start) & 
                    (df['PurchaseDate'].dt.date <= today)
                ]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for the last fortnight (from {fortnight_start} to {today}) is ${fortnight_profit_margin:.2f}."
                )
                return []
            if last_months:
                logging.info("profit margin of last  months")
                try:
                    n_months = int(last_months_requested)
                    start_date = pd.to_datetime('today') - pd.DateOffset(months=n_months)
                    last_months_profit_margin = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= today)
                    ]['ProfitMargin'].sum()
                    dispatcher.utter_message(
                        text=f"The profit margin for the last {n_months} months (from {start_date.date()} to {today}) is ${last_months_profit_margin:.2f}."
                    )
                    return []
                except Exception:
                    dispatcher.utter_message(
                        text="Could not process the number of last months. Please provide a valid number."
                    )
                    return []

            if quarterly:
                logging.info("profit margin of quarterly")
                # Normalize user input to lowercase and remove spaces or numbers
                normalized_quarter = quarterly_requested.lower().replace(" ", "").replace("quarter", "")
                
                # Map quarters to start and end months
                current_year = pd.to_datetime('today').year
                quarters = {
                    "q1": (1, 3),
                    "q2": (4, 6),
                    "q3": (7, 9),
                    "q4": (10, 12)
                }
            
                if normalized_quarter in quarters:
                    start_month, end_month = quarters[normalized_quarter]
                    quarterly_profit_margin = df[
                        (df['PurchaseDate'].dt.month >= start_month) & 
                        (df['PurchaseDate'].dt.month <= end_month) & 
                        (df['PurchaseDate'].dt.year == current_year)
                    ]['ProfitMargin'].sum()
                    
                    dispatcher.utter_message(
                        text=f"The profit margin for {normalized_quarter.upper()} {current_year} is ${quarterly_profit_margin:.2f}."
                    )
                    return []

            if half_year:
                logging.info("profit margin of half yearly")
                current_month = pd.to_datetime('today').month
                current_year = pd.to_datetime('today').year
                if current_month <= 6:
                    start_month, end_month = 1, 6
                else:
                    start_month, end_month = 7, 12

                half_yearly_profit_margin = df[
                    (df['PurchaseDate'].dt.month >= start_month) & 
                    (df['PurchaseDate'].dt.month <= end_month) & 
                    (df['PurchaseDate'].dt.year == current_year)
                ]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for the last half-year ({start_month}/{current_year} to {end_month}/{current_year}) is ${half_yearly_profit_margin:.2f}."
                )
                return []


            # Handle month-year pairs
            if month_year_pairs:
                logging.info("profit margin of month year")
                try:
                    for month, year in month_year_pairs:
                        monthly_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == year)
                        ]['ProfitMargin'].sum()
                        dispatcher.utter_message(
                            text=f"The profit margin for {months_reverse[month]} {year} is ${monthly_profit_margin:.2f}."
                        )
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly profit margins: {str(e)}")
                return []

            # Handle years
            if years:
                logging.info("profit margin of year")
                try:
                    yearly_profit_margin = df[df['PurchaseDate'].dt.year.isin(years)]['ProfitMargin'].sum()
                    years_str = ', '.join(map(str, years))
                    dispatcher.utter_message(
                        text=f"The total profit margin for {years_str} is ${yearly_profit_margin:.2f}."
                    )
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing yearly profit margins: {str(e)}")
                return []

            # Handle months in the current year
            if months_extracted:
                logging.info("profit margin of month with current year")
                current_year = pd.to_datetime('today').year
                try:
                    for month in months_extracted:
                        monthly_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == current_year)
                        ]['ProfitMargin'].sum()
                        dispatcher.utter_message(
                            text=f"The profit margin for {months_reverse[month]} {current_year} is ${monthly_profit_margin:.2f}."
                        )
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly profit margins: {str(e)}")
                return []

            # Handle total profit margin
            logging.info("total profit margin")
            total_profit_margin = df['ProfitMargin'].sum()
            dispatcher.utter_message(
                text=f"The overall total profit margin is ${total_profit_margin:.2f}."
            )

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while processing your request: {str(e)}")
        
        return []
