from docplex.cp.model import CpoModel
from collections import namedtuple
import numpy as np
TIME_LIMIT = 100

Company = namedtuple('company', ('name', 'cost', 'extra'))
Facility = namedtuple(
    'facility', ('name', 'type', 'company', 'content', 'extra'))
Customer = namedtuple(
    'customer', ('name', 'type', 'company', 'content', 'extra'))
FacilityTypeDemand = namedtuple(
    'facilityTypeDemand', ('type', 'demand', 'extra'))


def solve_seer_ilp(companies, facilities, customers, facilityDemands, costs, alpha=1, log_verbose='Quiet'):
    assert alpha >= 0  # ensure trace-off factor is >= 0, which show the contribution of representative cost
    assert log_verbose in ['Quiet', 'Terse', 'Normal', 'Verbose']
    n_company = len(companies)
    n_facility = len(facilities)
    n_customer = len(customers)
    n_type = len(facilityDemands)

    mdl = CpoModel()

    customerSupplier = mdl.integer_var_list(
        n_customer, 0, n_facility - 1, 'customerSupplier')
    openFacility = mdl.integer_var_list(n_facility, 0, 1, 'openFacility')
    facilityType = mdl.integer_var_list(n_facility, 0, n_type, 'facilityType')
    customerType = mdl.integer_var_list(
        n_customer, 0, n_type - 1, 'customerType')
    openCompany = mdl.integer_var_list(n_company, 0, 1, 'openCompany')
    company2id = {companies[i].name: i for i in range(n_company)}
    type2id = {facilityDemands[i].type: i for i in range(n_type)}
    facilityCompany = [company2id[facilities[i].company]
                       for i in range(n_facility)]

    validFacilityType = np.zeros((n_facility, n_type + 1))
    validFacilityType[:, n_type] = 1
    for i in range(n_facility):
        validFacilityType[i, type2id[facilities[i].type]] = 1

    for i in range(n_customer):
        mdl.add(mdl.element(openFacility, customerSupplier[i]) == 1)
        mdl.add(mdl.element(customerType, i) == mdl.element(
            facilityType, customerSupplier[i]))
        mdl.add(mdl.element(mdl.element(customerType, i),
                            validFacilityType[i]) == 1)

    for i in range(n_facility):
        mdl.add(mdl.element(mdl.element(facilityType, i),
                            validFacilityType[i]) == 1)
        mdl.add(mdl.element(openFacility, i) <=
                mdl.element(openCompany, facilityCompany[i]))
        mdl.add((mdl.element(openFacility, i) == 1) ==
                (mdl.element(facilityType, i) < n_type))

    for i in range(n_type):
        mdl.add(mdl.count(facilityType, i) == facilityDemands[i].demand)

    # Objective
    total_cost = mdl.scal_prod(
        openCompany, [c.cost for c in companies])
    for i in range(n_customer):
        total_cost += mdl.element(customerSupplier[i], alpha * costs[i])
    mdl.add(mdl.minimize(total_cost))

    # -----------------------------------------------------------------------------
    # Solve the model and display the result
    # -----------------------------------------------------------------------------

    # Solve model
    msol = mdl.solve(TimeLimit=TIME_LIMIT, LogVerbosity=log_verbose)

    selectedFacilities = []
    selectedCompanies = []
    if msol:
        for i in range(n_facility):
            if msol[facilityType[i]] < n_type:
                selectedFacilities.append(i)
                if facilityCompany[i] not in selectedCompanies:
                    selectedCompanies.append(facilityCompany[i])
    return msol, selectedFacilities, selectedCompanies


def main():
    companies = [
        Company('review1', 0.01, None),
        Company('review2', 0.02, None),
        Company('review3', 0.03, None)
    ]

    facilities = [
        Facility('s1', 'battery', 'review1',
                 'The battery life of this phone is good', None),
        Facility('s2', 'battery', 'review2',
                 'This phone has very long battery life', None),
        Facility('s3', 'battery', 'review3',
                 'The phone battery is not bad', None),
        Facility('s4', 'screen', 'review1',
                 'The screen of this phone is wide', None),
        Facility('s5', 'screen', 'review2',
                 'The phone\'s screen is fancy', None),
        Facility('s6', 'screen', 'review3', 'I love the big screen', None),
        Facility('s7', 'screen', 'review1', 'This big screen is a plus', None),
        Facility('s8', 'cover', 'review2', '...', None)
    ]

    customers = [
        Customer('s1', 'battery', 'review1',
                 'The battery life of this phone is good', None),
        Customer('s2', 'battery', 'review2',
                 'This phone has very long battery life', None),
        Customer('s3', 'battery', 'review3',
                 'The phone battery is not bad', None),
        Customer('s4', 'screen', 'review1',
                 'The screen of this phone is wide', None),
        Customer('s5', 'screen', 'review2',
                 'The phone\'s screen is fancy', None),
        Customer('s6', 'screen', 'review3', 'I love the big screen', None),
        Customer('s7', 'screen', 'review1', 'This big screen is a plus', None),
        Customer('s8', 'cover', 'review2', '...', None)
    ]

    facilityDemands = [
        FacilityTypeDemand('battery', 2, None),
        FacilityTypeDemand('screen', 2, None),
        FacilityTypeDemand('cover', 1, None),
    ]

    costs = np.array([
        [0.01, 1, 0.5, 0.5, 1, 1, 1, 1],
        [1, 0.02, 1, 1, 1, 1, 1, 1],
        [0.5, 1, 0.03, 1, 1, 1, 1, 1],
        [0.5, 1, 1, 0.015, 0.5, 1, 1, 1],
        [1, 1, 1, 0.5, 0.02, 1, 1, 1],
        [1, 1, 1, 1, 1, 0.03, 1, 1],
        [1, 1, 1, 1, 1, 1, 0.01, 1],
        [1, 1, 1, 1, 1, 1, 1, 0.02]
    ])

    solution, selectedFacilities, selectedCompanies = solve_seer_ilp(
        companies, facilities, customers, facilityDemands, costs, alpha=10)
    print(selectedCompanies, selectedFacilities)


if __name__ == '__main__':
    main()
