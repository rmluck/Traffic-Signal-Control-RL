import cityflow

def run_simulation():
    config_file = "configs/basic/basic_config.json"
    engine = cityflow.Engine(config_file, thread_num=1)

    for step in range(100):
        engine.next_step()
        print(f"Step: {step}, Vehicle Count: {engine.get_vehicle_count()}")
    
    engine.close()

if __name__ == "__main__":

    run_simulation()