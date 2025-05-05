# -*- coding: utf-8 -*-
import streamlit as st
import os
import requests
from typing import Type
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchResults
from langchain.schema import SystemMessage

# "gpt-3.5-turbo" "gpt-4o-mini"
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

alpha_vantage_api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")


class StockMarketSymbolSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")


class StockMarketSymbolSearchTool(BaseTool):
    name: str = "StockMarketSymbolSearchTool"
    description: str = """
  Use this tool to find the stock market symbol for a given company.
  It takes a query as an argument.
  Example query: Stock Market Symbol for Apple Company
  """
    args_schema: Type[StockMarketSymbolSearchToolArgsSchema] = StockMarketSymbolSearchToolArgsSchema

    def _run(self, query):
        ddg = DuckDuckGoSearchResults()
        return ddg.run(query)


class CompanyOverviewToolArgsSchema(BaseModel):
    symbol: str = Field(
        description="Stock symbol of the company. Exaple: AAPL,TSLA")


class CompanyOverviewTool(BaseTool):
    name: str = "CompanyOverview"
    description: str = """
  Use this to get an overview of the financiaals of the company.
  You should enter a stock symbol.
  """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()


class CompanyIncomeStatementTool(BaseTool):
    name: str = "CompanyIncomeStatement"
    description: str = """
  Use this to get an income statement of the financiaals of the company.
  You should enter a stock symbol.
  """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={symbol}&apikey={alpha_vantage_api_key}")
        return r.json()["annualReports"]


class CompanyStockPerformanceTool(BaseTool):
    name: str = "CompanyStockPerformance"
    description: str = """
  Use this to get the weekly performance of a company stock.
  You should enter a stock symbol.
  """
    args_schema: Type[CompanyOverviewToolArgsSchema] = CompanyOverviewToolArgsSchema

    def _run(self, symbol):
        r = requests.get(
            f"https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={symbol}&apikey={alpha_vantage_api_key}")
        response = r.json()
        return list(response["Weekly Time Series"].keys())[:200]


agent = initialize_agent(
    llm=llm,
    verbose=True,
    agent=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
    tools=[
        StockMarketSymbolSearchTool(),
        CompanyOverviewTool(),
        CompanyIncomeStatementTool(),
        CompanyStockPerformanceTool(),
    ],
    agent_kwargs={
        "system_message": SystemMessage(content="""
            Your are a hedge fund manager.

            You evaluate a company and provide your opinion and reasons why the stock is a buy or not.

            Consider the preformance of a stock, the company overview and the income statement.

            Be assertive ain your judgement and recommend the stock or advise the user against it.
        """),
    }
)


st.set_page_config(
    page_title="InvestorGPT",
    page_icon="📈",
)


st.markdown(
    """
  # InvestorGPT

  Welcome to InvestorGPT.

  Write down the name of a compay and our Agent will do the research for you.
  """
)

company = st.text_input("Enter the name of the company")

if company:
    result = agent.invoke(company)
    st.write(result["output"])
