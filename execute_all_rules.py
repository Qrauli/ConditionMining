from ruleforge.main_func import run_ruleforges_of_various_rule_types
import os

datasets = [
    ('Ecommerce-500000', "examples/datasets/Ecommerce-500000.csv"),
    #('Ecommerce-10000', "examples/datasets/Ecommerce-10000.csv"),
    #('Ecommerce-1000000', "examples/datasets/Ecommerce-1000000.csv"),
    #('Ecommerce', "examples/datasets/Ecommerce.csv"),
    #('SupplyChain', "examples/datasets/SupplyChain.csv"),
    #('University', "examples/datasets/University.csv"),
    #('RealEstate', "examples/datasets/RealEstate.csv"),
    #('Insurance', "examples/datasets/Insurance.csv"),
    #('Manufacturing', "examples/datasets/Manufacturing.csv"),
    #('Airline', "examples/datasets/Airline.csv"),
    #('Healthcare', "examples/datasets/Healthcare.csv"),
    #('Banking', "examples/datasets/Banking.csv"),
    #('HR', "examples/datasets/HR.csv"),

    ]


for dataset_name, file_path in datasets:
    run_ruleforges_of_various_rule_types(
        dataset_name=dataset_name,
        file_path=file_path,
        cache_enabled=True,
        skip_scoring=False
    )

