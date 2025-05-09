import pandas as pd
import numpy as np
import ast
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# Function to parse the CSV files
def load_data():
    """Load and preprocess the job matching data"""
    # Load the data
    seekers_df = pd.read_csv('seekers.csv')
    jobs_df = pd.read_csv('jobs.csv')
    distances_df = pd.read_csv('location_distances.csv', index_col=0)
    
    # Convert string representations to actual lists
    seekers_df['Skills'] = seekers_df['Skills'].apply(ast.literal_eval)
    seekers_df['Questionnaire'] = seekers_df['Questionnaire'].apply(ast.literal_eval)
    jobs_df['Required_Skills'] = jobs_df['Required_Skills'].apply(ast.literal_eval)
    jobs_df['Questionnaire'] = jobs_df['Questionnaire'].apply(ast.literal_eval)
    
    # Convert numeric fields to the appropriate types
    seekers_df['Min_Desired_Salary'] = seekers_df['Min_Desired_Salary'].astype(int)
    seekers_df['Max_Commute_Distance'] = seekers_df['Max_Commute_Distance'].astype(int)
    jobs_df['Is_Remote'] = jobs_df['Is_Remote'].astype(int)
    jobs_df['Salary_Range_Min'] = jobs_df['Salary_Range_Min'].astype(int)
    jobs_df['Salary_Range_Max'] = jobs_df['Salary_Range_Max'].astype(int)
    jobs_df['Num_Positions'] = jobs_df['Num_Positions'].astype(int)
    jobs_df['Priority_Weight'] = jobs_df['Priority_Weight'].astype(int)
    
    return seekers_df, jobs_df, distances_df

# Define the experience level order
def get_experience_level_order():
    return {
        'Entry-level': 1,
        'Mid-level': 2,
        'Senior': 3,
        'Lead': 4,
        'Manager': 5
    }

# Function to check compatibility between a seeker and a job
def check_compatibility(seeker, job, distances_df):
    """
    Check if a seeker is compatible with a job based on the required criteria
    Returns: True if compatible, False otherwise
    """
    # Experience level compatibility
    exp_order = get_experience_level_order()
    seeker_exp = exp_order.get(seeker['Experience_Level'], 0)
    job_exp = exp_order.get(job['Required_Experience_Level'], 6)  # Default high if unknown
    if seeker_exp < job_exp:
        return False
    
    # Job type compatibility
    if seeker['Desired_Job_Type'] != job['Job_Type']:
        return False
    
    # Salary compatibility
    if seeker['Min_Desired_Salary'] > job['Salary_Range_Max']:
        return False
    
    # Skills compatibility
    if not all(skill in seeker['Skills'] for skill in job['Required_Skills']):
        return False
    
    # Location compatibility
    if job['Is_Remote'] == 1:
        # Remote jobs are always compatible
        return True
    else:
        # Non-remote: check if the distance is within the seeker's max commute distance
        seeker_location = seeker['Location']
        job_location = job['Location']
        distance = distances_df.loc[seeker_location, job_location]
        return distance <= seeker['Max_Commute_Distance']

# Function to calculate dissimilarity scores between seekers and jobs
def calculate_dissimilarity(seeker, job):
    """Calculate the dissimilarity score between a seeker and a job based on questionnaire responses"""
    total_diff = sum(abs(sq - jq) for sq, jq in zip(seeker['Questionnaire'], job['Questionnaire']))
    return total_diff / 20  # Average difference across 20 questions

# Part 1: Maximize Priority-Weighted Matches
def solve_part1(seekers_df, jobs_df, distances_df):
    """
    Solve the first ILP model to maximize the sum of priority weights
    Returns: The model, compatible pairs dict, and the maximum priority weight
    """
    print("Solving Part 1: Maximize Priority-Weighted Matches")
    
    # Find all compatible seeker-job pairs
    compatible_pairs = {}
    for _, seeker in seekers_df.iterrows():
        for _, job in jobs_df.iterrows():
            if check_compatibility(seeker, job, distances_df):
                seeker_id = seeker['Seeker_ID']
                job_id = job['Job_ID']
                if seeker_id not in compatible_pairs:
                    compatible_pairs[seeker_id] = []
                compatible_pairs[seeker_id].append(job_id)
    
    # Create optimization model
    model = gp.Model("JobMatching_Part1")
    
    # Decision variables: x[i,j] = 1 if seeker i is assigned to job j, 0 otherwise
    x = {}
    for seeker_id, job_ids in compatible_pairs.items():
        for job_id in job_ids:
            x[seeker_id, job_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{seeker_id}_{job_id}")
    
    # Objective: Maximize sum of priority weights
    obj = gp.LinExpr()
    for (seeker_id, job_id), var in x.items():
        job_weight = jobs_df.loc[jobs_df['Job_ID'] == job_id, 'Priority_Weight'].values[0]
        obj += job_weight * var
    
    model.setObjective(obj, GRB.MAXIMIZE)
    
    # Constraint 1: Each seeker can be assigned to at most one job
    for seeker_id in compatible_pairs.keys():
        if seeker_id in compatible_pairs:
            model.addConstr(
                gp.quicksum(x[seeker_id, job_id] for job_id in compatible_pairs[seeker_id]) <= 1,
                f"seeker_constraint_{seeker_id}"
            )
    
    # Constraint 2: Number of seekers assigned to a job cannot exceed its positions
    job_to_seeker = {}
    for seeker_id, job_ids in compatible_pairs.items():
        for job_id in job_ids:
            if job_id not in job_to_seeker:
                job_to_seeker[job_id] = []
            job_to_seeker[job_id].append(seeker_id)
    
    for job_id, seeker_ids in job_to_seeker.items():
        job_positions = jobs_df.loc[jobs_df['Job_ID'] == job_id, 'Num_Positions'].values[0]
        model.addConstr(
            gp.quicksum(x[seeker_id, job_id] for seeker_id in seeker_ids) <= job_positions,
            f"job_constraint_{job_id}"
        )
    
    # Solve the model
    model.optimize()
    
    # Extract the maximum priority weight
    max_priority_weight = model.objVal
    
    print(f"Maximum Priority Weight (Mw): {max_priority_weight}")
    
    return model, compatible_pairs, max_priority_weight

# Part 2: Minimize Maximum Dissimilarity
def solve_part2(seekers_df, jobs_df, distances_df, compatible_pairs, max_priority_weight, omega):
    """
    Solve the second ILP model to minimize the maximum dissimilarity
    while maintaining at least omega% of the maximum priority weight
    
    Parameters:
    - omega: Percentage of maximum priority weight to maintain (0-100)
    
    Returns: The model and the minimum maximum dissimilarity
    """
    print(f"\nSolving Part 2: Minimize Maximum Dissimilarity (ω = {omega}%)")
    
    # Calculate dissimilarity scores for all compatible pairs
    dissimilarity_scores = {}
    for seeker_id, job_ids in compatible_pairs.items():
        seeker = seekers_df.loc[seekers_df['Seeker_ID'] == seeker_id].iloc[0]
        for job_id in job_ids:
            job = jobs_df.loc[jobs_df['Job_ID'] == job_id].iloc[0]
            dissimilarity_scores[seeker_id, job_id] = calculate_dissimilarity(seeker, job)
    
    # Create optimization model
    model = gp.Model("JobMatching_Part2")
    
    # Decision variables: x[i,j] = 1 if seeker i is assigned to job j, 0 otherwise
    x = {}
    for seeker_id, job_ids in compatible_pairs.items():
        for job_id in job_ids:
            x[seeker_id, job_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{seeker_id}_{job_id}")
    
    # Additional variable for the maximum dissimilarity
    max_dissimilarity = model.addVar(vtype=GRB.CONTINUOUS, name="max_dissimilarity")
    
    # Objective: Minimize the maximum dissimilarity
    model.setObjective(max_dissimilarity, GRB.MINIMIZE)
    
    # Constraint: Maximum dissimilarity
    for (seeker_id, job_id), score in dissimilarity_scores.items():
        model.addConstr(
            max_dissimilarity >= score * x[seeker_id, job_id],
            f"max_dissimilarity_{seeker_id}_{job_id}"
        )
    
    # Constraint 1: Each seeker can be assigned to at most one job
    for seeker_id in compatible_pairs.keys():
        if seeker_id in compatible_pairs:
            model.addConstr(
                gp.quicksum(x[seeker_id, job_id] for job_id in compatible_pairs[seeker_id]) <= 1,
                f"seeker_constraint_{seeker_id}"
            )
    
    # Constraint 2: Number of seekers assigned to a job cannot exceed its positions
    job_to_seeker = {}
    for seeker_id, job_ids in compatible_pairs.items():
        for job_id in job_ids:
            if job_id not in job_to_seeker:
                job_to_seeker[job_id] = []
            job_to_seeker[job_id].append(seeker_id)
    
    for job_id, seeker_ids in job_to_seeker.items():
        job_positions = jobs_df.loc[jobs_df['Job_ID'] == job_id, 'Num_Positions'].values[0]
        model.addConstr(
            gp.quicksum(x[seeker_id, job_id] for seeker_id in seeker_ids) <= job_positions,
            f"job_constraint_{job_id}"
        )
    
    # Constraint 3: Maintain at least omega% of the maximum priority weight
    priority_expr = gp.LinExpr()
    for (seeker_id, job_id), var in x.items():
        job_weight = jobs_df.loc[jobs_df['Job_ID'] == job_id, 'Priority_Weight'].values[0]
        priority_expr += job_weight * var
    
    min_required_weight = (omega / 100) * max_priority_weight
    model.addConstr(
        priority_expr >= min_required_weight,
        "min_priority_weight"
    )
    
    # Solve the model
    model.optimize()
    
    # Extract the minimum maximum dissimilarity
    min_max_dissimilarity = model.objVal if model.status == GRB.OPTIMAL else None
    achieved_weight = priority_expr.getValue() if model.status == GRB.OPTIMAL else None
    
    print(f"Minimum Maximum Dissimilarity: {min_max_dissimilarity}")
    print(f"Achieved Priority Weight: {achieved_weight} ({achieved_weight/max_priority_weight*100:.2f}% of maximum)")
    
    return model, min_max_dissimilarity, achieved_weight

# Function to run the entire solution
def run_job_matching_solution():
    # Load data
    seekers_df, jobs_df, distances_df = load_data()
    
    # Part 1: Maximize Priority-Weighted Matches
    model_p1, compatible_pairs, max_priority_weight = solve_part1(seekers_df, jobs_df, distances_df)
    
    # Part 2: Minimize Maximum Dissimilarity for different omega values
    omega_values = [70, 75, 80, 85, 90, 95, 100]
    results = []
    
    for omega in omega_values:
        model_p2, min_max_dissimilarity, achieved_weight = solve_part2(
            seekers_df, jobs_df, distances_df, compatible_pairs, max_priority_weight, omega
        )
        results.append({
            'omega': omega,
            'min_max_dissimilarity': min_max_dissimilarity,
            'achieved_weight': achieved_weight,
            'achieved_percentage': achieved_weight/max_priority_weight*100
        })
    
    # Create visualization of the results
    plot_results(results, max_priority_weight)
    
    return results

# Function to visualize the results
def plot_results(results, max_priority_weight):
    """Create a plot showing how dissimilarity changes with omega"""
    # Extract data for plotting
    omegas = [r['omega'] for r in results]
    dissimilarities = [r['min_max_dissimilarity'] for r in results]
    
    # Create the figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Primary y-axis: Minimum Maximum Dissimilarity
    color = 'tab:blue'
    ax1.set_xlabel('ω (% of Maximum Priority Weight)')
    ax1.set_ylabel('Minimum Maximum Dissimilarity', color=color)
    ax1.plot(omegas, dissimilarities, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Add data labels
    for i, txt in enumerate(dissimilarities):
        ax1.annotate(f'{txt:.3f}', 
                    (omegas[i], dissimilarities[i]),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Set title and adjust layout
    plt.title('Trade-off between Priority Weight (ω) and Minimum Maximum Dissimilarity')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('omega_dissimilarity_tradeoff.png')
    plt.close()

# Function to analyze and compare the solutions
def analyze_solutions(results, max_priority_weight):
    """Analyze the results and recommend an omega value"""
    # Calculate the percentage increase in dissimilarity
    base_dissimilarity = results[-1]['min_max_dissimilarity']  # at ω = 100%
    
    for r in results:
        if base_dissimilarity > 0:
            r['dissimilarity_increase'] = (r['min_max_dissimilarity'] - base_dissimilarity) / base_dissimilarity * 100
        else:
            r['dissimilarity_increase'] = 0
    
    # Find the "elbow point" where diminishing returns start
    # Use a simple heuristic: find where the rate of change significantly increases
    dissimilarity_changes = []
    for i in range(1, len(results)):
        change = results[i]['min_max_dissimilarity'] - results[i-1]['min_max_dissimilarity']
        dissimilarity_changes.append(change)
    
    # Find the largest negative change (where dissimilarity drops most)
    if dissimilarity_changes:
        largest_change_idx = dissimilarity_changes.index(min(dissimilarity_changes)) + 1
        recommended_omega = results[largest_change_idx]['omega']
    else:
        recommended_omega = results[-1]['omega']  # Default to 100% if can't find elbow
    
    print("\n=== SOLUTION ANALYSIS ===")
    print(f"Maximum Priority Weight (Mw): {max_priority_weight}")
    print("\nResults for different ω values:")
    for r in results:
        print(f"ω = {r['omega']}%: Max Dissimilarity = {r['min_max_dissimilarity']:.3f}, "
              f"Weight = {r['achieved_weight']:.1f} ({r['achieved_percentage']:.1f}% of max)")
    
    print(f"\nRecommended ω value: {recommended_omega}%")
    print("Justification: This value offers a good balance between minimizing dissimilarity and maintaining a reasonable priority weight.")
    
    return recommended_omega

# Main execution function
def main():
    """Main function to execute the entire solution"""
    print("=== JOBSYNC OPTIMIZATION SOLUTION ===")
    
    # Run the solution
    results = run_job_matching_solution()
    
    # Analyze the results and recommend omega
    max_priority_weight = results[0]['achieved_weight'] * 100 / results[0]['achieved_percentage']
    recommended_omega = analyze_solutions(results, max_priority_weight)
    
    print("\nOptimization complete. Check 'omega_dissimilarity_tradeoff.png' for the visualization.")

if __name__ == "__main__":
    main()