import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from concurrent.futures import ThreadPoolExecutor

# Function to visualize sorting steps with color changes
def visualize_sorting(arr, colors, algorithm_name, ax, fig):
    ax.clear()  # Clear the previous plot
    ax.bar(range(len(arr)), arr, color=colors)

    # Displaying the numbers on top of the bars
    for i in range(len(arr)):
        ax.text(i, arr[i] + 0.2, str(arr[i]), ha='center', fontsize=10)

    ax.set_title(f"Sorting Visualization: {algorithm_name}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Value")
    st.pyplot(fig)

# Bubble Sort Algorithm with Animation
def bubble_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            colors = ['skyblue'] * n
            colors[j] = 'red'  # Highlight element being compared
            colors[j + 1] = 'red'  # Highlight the other element being compared
            comparisons += 1

            if (ascending and arr[j] > arr[j + 1]) or (not ascending and arr[j] < arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                colors[j] = 'green'  # Highlight sorted element
                colors[j + 1] = 'green'  # Highlight sorted element
                swaps += 1

            steps.append((arr[:], colors))  # Store the current step and colors
            explanations.append(f"Compared {arr[j]} and {arr[j + 1]}, swapped them.")
    return steps, explanations, comparisons, swaps

# Insertion Sort Algorithm with Animation
def insertion_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        colors = ['skyblue'] * len(arr)
        colors[i] = 'red'  # Highlight element being inserted
        while j >= 0 and ((ascending and key < arr[j]) or (not ascending and key > arr[j])):
            comparisons += 1
            arr[j + 1] = arr[j]
            j -= 1
            colors[j + 1] = 'green'  # Highlight the sorted position
            swaps += 1
        arr[j + 1] = key
        colors[j + 1] = 'green'  # Final sorted position

        steps.append((arr[:], colors))  # Store the current step and colors
        explanations.append(f"Inserted {key} into the correct position.")
    return steps, explanations, comparisons, swaps

# Quick Sort Algorithm with Animation
def quick_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0

    def _quick_sort(arr, low, high):
        if low < high:
            pi = partition(arr, low, high)
            _quick_sort(arr, low, pi - 1)
            _quick_sort(arr, pi + 1, high)

    def partition(arr, low, high):
        nonlocal comparisons, swaps
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            colors = ['skyblue'] * len(arr)
            colors[j] = 'red'  # Highlight the element being compared
            colors[high] = 'green'  # Highlight pivot element
            comparisons += 1

            if (ascending and arr[j] < pivot) or (not ascending and arr[j] > pivot):
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                colors[i] = 'green'  # Element in final sorted position
                colors[j] = 'green'  # Element in final sorted position
                swaps += 1
            steps.append((arr[:], colors))  # Store the current step and colors

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        explanations.append(f"Pivot {pivot} placed at correct position.")
        return i + 1

    _quick_sort(arr, 0, len(arr) - 1)
    return steps, explanations, comparisons, swaps

# Selection Sort Algorithm with Animation
def selection_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            colors = ['skyblue'] * n
            colors[i] = 'red'  # Highlight the current position of the minimum element
            colors[j] = 'red'  # Highlight the element being compared
            comparisons += 1
            if (ascending and arr[j] < arr[min_idx]) or (not ascending and arr[j] > arr[min_idx]):
                min_idx = j
            steps.append((arr[:], colors))  # Store the current step and colors
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        colors[i] = 'green'  # Highlight the sorted element
        steps.append((arr[:], colors))  # Store the current step and colors
        explanations.append(f"Swapped {arr[i]} with {arr[min_idx]}.")
        swaps += 1
    return steps, explanations, comparisons, swaps

# Radix Sort Algorithm with Animation
def radix_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0

    def counting_sort(arr, exp, ascending):
        nonlocal comparisons, swaps
        n = len(arr)
        output = [0] * n
        count = [0] * 10
        for i in range(n):
            index = arr[i] // exp
            count[index % 10] += 1
        for i in range(1, 10):
            count[i] += count[i - 1]
        i = n - 1
        while i >= 0:
            index = arr[i] // exp
            output[count[index % 10] - 1] = arr[i]
            count[index % 10] -= 1
            i -= 1
        for i in range(n):
            arr[i] = output[i]
        explanations.append(f"Sorted by digit place {exp}.")

    max_elem = max(arr)
    exp = 1
    while max_elem // exp > 0:
        counting_sort(arr, exp, ascending)
        steps.append((arr[:], ['skyblue'] * len(arr)))
        exp *= 10
    return steps, explanations, comparisons, swaps

# Merge Sort Algorithm with Animation
def merge_sort(arr, ascending=True):
    steps = []
    explanations = []  # To store explanation steps
    comparisons = 0
    swaps = 0

    def _merge_sort(arr, left, right):
        if left < right:
            mid = (left + right) // 2
            _merge_sort(arr, left, mid)
            _merge_sort(arr, mid + 1, right)
            merge(arr, left, mid, right)

    def merge(arr, left, mid, right):
        nonlocal comparisons, swaps
        n1 = mid - left + 1
        n2 = right - mid
        L = arr[left:mid + 1]
        R = arr[mid + 1:right + 1]
        i = j = 0
        k = left
        while i < n1 and j < n2:
            colors = ['skyblue'] * len(arr)
            colors[left + i] = 'red'
            colors[mid + 1 + j] = 'red'
            comparisons += 1
            if (ascending and L[i] <= R[j]) or (not ascending and L[i] >= R[j]):
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
            steps.append((arr[:], colors))
            explanations.append(f"Merged {L[i - 1]} and {R[j - 1]}.")
            swaps += 1
        while i < n1:
            arr[k] = L[i]
            i += 1
            k += 1
        while j < n2:
            arr[k] = R[j]
            j += 1
            k += 1
        steps.append((arr[:], ['green'] * len(arr)))
        explanations.append("Merging complete.")

    _merge_sort(arr, 0, len(arr) - 1)
    return steps, explanations, comparisons, swaps

# Function to generate array from input string
def generate_array(arr_str):
    try:
        arr = list(map(int, arr_str.split()))
        return arr
    except ValueError:
        st.error("Please enter valid integers separated by spaces.")
        return []

# Function to generate random array based on input parameters
def generate_random_array(size, min_val, max_val):
    return np.random.randint(min_val, max_val + 1, size).tolist()

# Fun facts about sorting algorithms
def display_fun_fact():
    fun_facts = [
        "Bubble Sort is named because smaller elements 'bubble' to the top of the list.",
        "Quick Sort is one of the fastest sorting algorithms, with an average time complexity of O(n log n).",
        "Merge Sort is a stable sort, meaning it preserves the relative order of equal elements.",
        "Insertion Sort is efficient for small datasets and is often used in practice for small arrays.",
        "Selection Sort always performs O(n^2) comparisons, regardless of the input.",
        "Radix Sort is a non-comparative sorting algorithm that works on digits of numbers.",
        "The fastest sorting algorithms, like Quick Sort and Merge Sort, use a divide-and-conquer approach.",
    ]
    st.sidebar.markdown("### Fun Fact")
    st.sidebar.write(random.choice(fun_facts))

# Main function to handle user input and visualization
    # Main function to handle user input and visualization
def main():
    st.title("Sorting Algorithm Visualizer")

    # Sidebar for input options
    with st.sidebar:
        st.header("Input Options")
        # Option 1: User inputs array manually
        input_type = st.radio("Choose Input Type", ["Manual Input", "Random Array"])

        if input_type == "Manual Input":
            arr_str = st.text_input("Enter Array (space-separated):", "10 5 3 8 6 2")
            arr = generate_array(arr_str)

        if input_type == "Random Array":
            size = st.number_input("Array Size:", min_value=1, value=20)
            min_val = st.number_input("Min Value:", min_value=1, value=10)
            max_val = st.number_input("Max Value:", min_value=1, value=100)
            arr = generate_random_array(size, min_val, max_val)

        # Ascending/Descending Option
        order = st.radio("Choose Sorting Order", ['Ascending', 'Descending'])
        ascending = order == 'Ascending'

        # Sorting Algorithm Choice
        algorithm = st.selectbox(
            "Choose Sorting Algorithm",
            ["Bubble Sort", "Insertion Sort", "Quick Sort", "Selection Sort", "Radix Sort", "Merge Sort", "Sorting Race 🏁"]
        )

        # Additional input for Insertion Sort
        if algorithm == "Insertion Sort":
            new_element = st.text_input("Enter a new element to add to the array:")
            if new_element:
                try:
                    new_element = int(new_element)
                    arr.append(new_element)
                    st.write(f"Added new element: {new_element}")
                except ValueError:
                    st.error("Please enter a valid integer.")

        # Reset Button
        if st.button("Reset", key="reset_button"):
            st.session_state.clear()
            st.rerun()

        # Sidebar explanation of colors
        st.markdown("### Color Legend")
        st.markdown("- **Red**: Elements being compared or moved.")
        st.markdown("- **Green**: Elements in their final sorted position.")
        st.markdown("- **Skyblue**: Elements not currently being interacted with.")

    # Display a random fun fact
    display_fun_fact()

    # Set up figure for plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Perform Sorting and Visualization
    if st.button("Start Sorting", key="start_sorting_button"):
        if algorithm == "Bubble Sort":
            steps, explanations, comparisons, swaps = bubble_sort(arr[:], ascending)
        elif algorithm == "Insertion Sort":
            steps, explanations, comparisons, swaps = insertion_sort(arr[:], ascending)
        elif algorithm == "Quick Sort":
            steps, explanations, comparisons, swaps = quick_sort(arr[:], ascending)
        elif algorithm == "Selection Sort":
            steps, explanations, comparisons, swaps = selection_sort(arr[:], ascending)
        elif algorithm == "Radix Sort":
            steps, explanations, comparisons, swaps = radix_sort(arr[:], ascending)
        elif algorithm == "Merge Sort":
            steps, explanations, comparisons, swaps = merge_sort(arr[:], ascending)
        elif algorithm == "Sorting Race 🏁":
            # Initialize a dictionary to store the results of each algorithm
            race_results = {}

            # Run all sorting algorithms simultaneously
            algorithms = {
                "Bubble Sort": bubble_sort,
                "Insertion Sort": insertion_sort,
                "Quick Sort": quick_sort,
                "Selection Sort": selection_sort,
                "Radix Sort": radix_sort,
                "Merge Sort": merge_sort,
            }

            # Create a placeholder for the race results
            race_placeholder = st.empty()

            # Run each algorithm and measure the time taken
            for algo_name, algo_func in algorithms.items():
                start_time = time.time()
                steps, explanations, comparisons, swaps = algo_func(arr[:], ascending)
                end_time = time.time()
                race_results[algo_name] = end_time - start_time

                # Display the progress of each algorithm
                race_placeholder.write(f"{algo_name} completed in {race_results[algo_name]:.8f} seconds.")

            # Determine the winner
            winner = min(race_results, key=race_results.get)
            st.success(f"🏆 {winner} is the fastest with a time of {race_results[winner]:.8f} seconds!")

            # Display the results in a table
            st.write("### Sorting Race Results")
            st.table(race_results)

            return

        # Initialize session state for step tracking
        if "steps" not in st.session_state:
            st.session_state.steps = steps
            st.session_state.explanations = explanations
            st.session_state.step_index = 0
            st.session_state.comparisons = comparisons
            st.session_state.swaps = swaps

    # Display the current step if steps exist
    if "steps" in st.session_state:
        step_index = st.session_state.step_index
        steps = st.session_state.steps
        explanations = st.session_state.explanations
        comparisons = st.session_state.comparisons
        swaps = st.session_state.swaps

        if step_index < len(steps) and step_index < len(explanations):
            arr_step, colors = steps[step_index]
            visualize_sorting(arr_step, colors, algorithm, ax, fig)
            st.write(f"**Step {step_index + 1}:** {explanations[step_index]}")
            st.write(f"**Comparisons:** {comparisons}")
            st.write(f"**Swaps:** {swaps}")

            # Next Step Button
            if st.button("Next Step", key="next_step_button"):
                st.session_state.step_index += 1
                st.rerun()
        else:
            st.success("Sorting Complete!")
            visualize_sorting(steps[-1][0], ['green'] * len(steps[-1][0]), algorithm, ax, fig)
            st.write(f"**Final Sorted Array:** {steps[-1][0]}")
            st.write(f"**Total Comparisons:** {comparisons}")
            st.write(f"**Total Swaps:** {swaps}")

            # Display all steps (text only) at the end
            st.write("### All Steps:")
            for i, explanation in enumerate(explanations):
                st.write(f"**Step {i + 1}:** {explanation}")

            if st.button("Reset", key="reset_final_button"):
                st.session_state.clear()
                st.rerun()

if __name__ == "__main__":
    main()
