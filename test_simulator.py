from digital_twin.simulator import GridState, Generator, GridSimulator
from digital_twin.redispatch import RedispatchEngine
from digital_twin.metrics import calculate_resilience_score, get_risk_label

def main():
    print("--- DIGITAL TWIN SIMULATOR ENGINE TEST ---\n")
    
    # 1. Mock the Grid State (using numbers from Section 16 of the Document)
    # Total Demand: 31,420 MW
    # Total Generation: 33,180 MW
    # We will split this among a few mocked generators
    
    nuclear_plant = Generator(id="NUC_1", technology="Nuclear", installed_capacity_mw=1000.0, current_output_mw=1000.0)
    hydro_plant = Generator(id="HYD_1", technology="Hydro", installed_capacity_mw=2000.0, current_output_mw=800.0)
    gas_plant = Generator(id="GAS_1", technology="Gas", installed_capacity_mw=3000.0, current_output_mw=2000.0)
    wind_farm = Generator(id="WIND_1", technology="Wind", installed_capacity_mw=2500.0, current_output_mw=1200.0)
    
    mock_generators = {
        "NUC_1": nuclear_plant,
        "HYD_1": hydro_plant,
        "GAS_1": gas_plant,
        "WIND_1": wind_farm
    }
    
    # Create the baseline state
    # Total Gen = 1000 + 800 + 2000 + 1200 = 5000 MW (Using simplified numbers for the test)
    # Let's say Demand is 4900 MW
    initial_state = GridState(
        timestamp="2024-08-24T12:00:00",
        total_demand_mw=4900.0,
        generators=mock_generators
    )
    
    print(f"INITIAL STATE [12:00 PM]")
    print(f"Demand: {initial_state.total_demand_mw} MW")
    print(f"Generation: {initial_state.total_generation_mw} MW")
    print(f"Reserve Margin: {initial_state.reserve_capacity_mw} MW\n")
    
    # 2. Chaos Engineering: Inject a Failure
    # Simulate a major nuclear plant tripping offline
    print("[CHAOS EVENT INJECTED]: Nuclear Plant NUC_1 Trips Offline!")
    simulator = GridSimulator(initial_state)
    simulator.inject_generator_failure("NUC_1")
    
    post_failure_state = simulator.state
    deficit = post_failure_state.deficit_mw
    
    print(f"\nPOST-FAILURE STATE [12:01 PM]")
    print(f"Generation: {post_failure_state.total_generation_mw} MW")
    print(f"Immediate Deficit: {deficit} MW\n")
    
    # 3. Redispatch
    print("Running Redispatch Engine...")
    redispatcher = RedispatchEngine()
    result = redispatcher.execute_redispatch(post_failure_state)
    
    print(f"Redispatch Status: {result['status']}")
    for action, mw in result['actions'].items():
        print(f" -> Ramped up {action} by {mw} MW")
        
    print(f"\nFINAL STATE [12:15 PM]")
    print(f"Generation: {post_failure_state.total_generation_mw} MW")
    print(f"Reserve Margin: {post_failure_state.reserve_capacity_mw} MW")
    print(f"Unserved Energy: {result['unserved_mw']} MW\n")
    
    # 4. Resilience Metrics
    score = calculate_resilience_score(initial_state, post_failure_state, result['unserved_mw'])
    risk = get_risk_label(score)
    print(f"SYSTEM RESILIENCE SCORE: {score}/100")
    print(f"RISK LEVEL: {risk}")

if __name__ == "__main__":
    main()
