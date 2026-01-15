import os
import mysql.connector
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

class Database:
    def __init__(self):
        self.db_type = os.getenv("DB_TYPE", "mysql")
        self.host = os.getenv("DB_HOST", "localhost")
        self.user = os.getenv("DB_USER", "root")
        self.password = os.getenv("DB_PASS", "")
        self.database = os.getenv("DB_NAME", "onlinejudge")
        self.port = int(os.getenv("DB_PORT", 3306))
        self.conn = None

    def connect(self):
        print(f"Connecting to {self.db_type} database at {self.host}...")
        try:
            if self.db_type == "mysql":
                self.conn = mysql.connector.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    port=self.port
                )
            elif self.db_type == "postgres":
                self.conn = psycopg2.connect(
                    host=self.host,
                    user=self.user,
                    password=self.password,
                    dbname=self.database,
                    port=self.port
                )
            else:
                raise ValueError("Unsupported DB_TYPE. Use 'mysql' or 'postgres'.")
            print("Database connection successful.")
        except Exception as e:
            print(f"Failed to connect to database: {e}")
            raise

    def fetch_all_problems(self) -> List[Dict]:
        if not self.conn:
            self.connect()
        
        print("Executing fetching query...")
        cursor = self.conn.cursor(dictionary=True) if self.db_type == "mysql" else self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Adjust query based on actual schema, assuming 'id', 'code', 'name', 'description', 'pdf_url'
        query = "SELECT id, code, name, description, pdf_url FROM judge_problem"
        
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        print(f"Fetched {len(results)} rows from database.")
        return results

    def close(self):
        if self.conn:
            self.conn.close()
