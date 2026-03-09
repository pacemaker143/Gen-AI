"""
Advanced ML & NLP Assignment - Standout Implementation
======================================================
Author: Your Name
Date: February 2026

Features:
1. Gaussian Mixture Models (GMM) - Probabilistic Generative Model
2. Markov Chain Text Generator - Multiple orders support
3. Prompt Engineering with Gemini API:
   - Interview Approach
   - Chain of Thought (CoT)
   - Tree of Thought (ToT)
4. Zero-shot vs Few-shot Prompting Comparison
5. Performance Analytics & Visualizations
6. Export Results to JSON/CSV
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
from collections import defaultdict, Counter
import random
import json
import csv
from datetime import datetime
import time
import os
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: GAUSSIAN MIXTURE MODEL (GMM)
# ============================================================================

class EnhancedGMM:
    """
    Enhanced Gaussian Mixture Model with visualization and metrics
    """
    def __init__(self, n_components=3, n_samples=500):
        self.n_components = n_components
        self.n_samples = n_samples
        self.model = None
        self.X = None
        self.y_true = None
        self.y_pred = None
        self.metrics = {}
        
    def generate_data(self):
        """Generate synthetic data"""
        print(f"\n{'='*60}")
        print("GENERATING SYNTHETIC DATA FOR GMM")
        print(f"{'='*60}")
        
        self.X, self.y_true = make_blobs(
            n_samples=self.n_samples,
            centers=self.n_components,
            n_features=2,
            cluster_std=1.0,
            random_state=42
        )
        print(f"Generated {self.n_samples} samples with {self.n_components} clusters")
        return self.X, self.y_true
    
    def train_model(self):
        """Train GMM model"""
        print(f"\n{'='*60}")
        print("TRAINING GAUSSIAN MIXTURE MODEL")
        print(f"{'='*60}")
        
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42
        )
        
        start_time = time.time()
        self.model.fit(self.X)
        training_time = time.time() - start_time
        
        self.y_pred = self.model.predict(self.X)
        
        # Calculate metrics
        self.metrics = {
            'training_time': training_time,
            'bic': self.model.bic(self.X),
            'aic': self.model.aic(self.X),
            'log_likelihood': self.model.score(self.X) * len(self.X),
            'silhouette_score': silhouette_score(self.X, self.y_pred),
            'davies_bouldin_score': davies_bouldin_score(self.X, self.y_pred),
            'n_iterations': self.model.n_iter_,
            'converged': self.model.converged_
        }
        
        print(f"Model trained in {training_time:.4f} seconds")
        print(f"Converged: {self.metrics['converged']} (iterations: {self.metrics['n_iterations']})")
        print(f"BIC Score: {self.metrics['bic']:.2f}")
        print(f"AIC Score: {self.metrics['aic']:.2f}")
        print(f"Silhouette Score: {self.metrics['silhouette_score']:.4f}")
        
        return self.metrics
    
    def visualize(self, save_path='gmm_results.png'):
        """Create comprehensive visualization"""
        print(f"\n{'='*60}")
        print("CREATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        fig = plt.figure(figsize=(16, 5))
        
        # Plot 1: True Clusters
        ax1 = plt.subplot(131)
        scatter1 = ax1.scatter(self.X[:, 0], self.X[:, 1], 
                              c=self.y_true, cmap='viridis', 
                              alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.set_title('True Clusters', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        plt.colorbar(scatter1, ax=ax1)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GMM Predictions with Gaussian Contours
        ax2 = plt.subplot(132)
        scatter2 = ax2.scatter(self.X[:, 0], self.X[:, 1], 
                              c=self.y_pred, cmap='viridis', 
                              alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add Gaussian contours
        x_min, x_max = self.X[:, 0].min() - 2, self.X[:, 0].max() + 2
        y_min, y_max = self.X[:, 1].min() - 2, self.X[:, 1].max() + 2
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        Z = -self.model.score_samples(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        contour = ax2.contour(xx, yy, Z, levels=10, linewidths=1.5, 
                             colors='black', alpha=0.4)
        ax2.scatter(self.model.means_[:, 0], self.model.means_[:, 1], 
                   c='red', marker='X', s=300, edgecolors='black', 
                   linewidth=2, label='Centroids')
        ax2.set_title('GMM Predictions with Gaussians', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        plt.colorbar(scatter2, ax=ax2)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Selection Metrics
        ax3 = plt.subplot(133)
        metrics_data = {
            'BIC': abs(self.metrics['bic']),
            'AIC': abs(self.metrics['aic']),
            'Silhouette\n(scaled)': self.metrics['silhouette_score'] * 1000
        }
        bars = ax3.bar(metrics_data.keys(), metrics_data.values(), 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_title('Model Quality Metrics', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
        plt.close()
    
    def generate_new_samples(self, n_samples=50):
        """Generate new samples from learned distribution"""
        new_samples, labels = self.model.sample(n_samples)
        return new_samples, labels


# ============================================================================
# PART 2: MARKOV CHAIN TEXT GENERATOR
# ============================================================================

class MarkovChainGenerator:
    """
    Advanced Markov Chain Text Generator with multiple order support
    """
    def __init__(self, order=2):
        self.order = order
        self.chain = defaultdict(list)
        self.start_words = []
        
    def train(self, text):
        """Train Markov Chain on text corpus"""
        print(f"\n{'='*60}")
        print(f"TRAINING MARKOV CHAIN (Order {self.order})")
        print(f"{'='*60}")
        
        words = text.split()
        print(f"Corpus size: {len(words)} words")
        print(f"Unique words: {len(set(words))}")
        
        # Build chain
        for i in range(len(words) - self.order):
            state = tuple(words[i:i + self.order])
            next_word = words[i + self.order]
            self.chain[state].append(next_word)
            
            # Track sentence starts
            if i == 0 or words[i-1].endswith(('.', '!', '?')):
                self.start_words.append(state)
        
        print(f"Built chain with {len(self.chain)} states")
        print(f"Average transitions per state: {np.mean([len(v) for v in self.chain.values()]):.2f}")
        
    def generate(self, max_length=50, seed=None):
        """Generate text using Markov Chain"""
        if not self.chain:
            return "Model not trained!"
        
        # Choose starting state
        if seed:
            words = seed.split()
            if len(words) >= self.order:
                current = tuple(words[-self.order:])
            else:
                current = random.choice(self.start_words) if self.start_words else random.choice(list(self.chain.keys()))
        else:
            current = random.choice(self.start_words) if self.start_words else random.choice(list(self.chain.keys()))
        
        result = list(current)
        
        for _ in range(max_length - self.order):
            if current not in self.chain:
                break
            next_word = random.choice(self.chain[current])
            result.append(next_word)
            current = tuple(result[-self.order:])
        
        return ' '.join(result)
    
    def analyze_chain(self):
        """Analyze chain statistics"""
        stats = {
            'total_states': len(self.chain),
            'total_transitions': sum(len(v) for v in self.chain.values()),
            'avg_transitions_per_state': np.mean([len(v) for v in self.chain.values()]),
            'max_transitions': max(len(v) for v in self.chain.values()),
            'min_transitions': min(len(v) for v in self.chain.values())
        }
        return stats


# ============================================================================
# PART 3: PROMPT ENGINEERING WITH GEMINI API
# ============================================================================

class PromptEngineeringComparator:
    """
    Compare different prompt engineering approaches using Gemini API
    """
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.results = {}
        
        # For demo purposes without API
        self.demo_mode = api_key is None
        
    def interview_approach(self, problem):
        """Interview/Socratic Method Approach"""
        prompt = f"""Problem: {problem}

Let's solve this step by step through a series of questions:
1. What do we need to find?
2. What information do we have?
3. What steps should we take?
4. What is the solution?

Please answer each question and provide the final solution."""

        print(f"\n{'='*60}")
        print("INTERVIEW APPROACH")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}\n")
        
        response = self._call_api(prompt)
        
        return {
            'approach': 'Interview',
            'prompt': prompt,
            'response': response,
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
    
    def chain_of_thought(self, problem):
        """Chain of Thought (CoT) Approach"""
        prompt = f"""Problem: {problem}

Let's think through this step by step:
1. First, I'll identify what we're looking for
2. Then, I'll list the given information
3. Next, I'll work through the solution systematically
4. Finally, I'll verify the answer

Solution:"""

        print(f"\n{'='*60}")
        print("CHAIN OF THOUGHT (CoT)")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}\n")
        
        response = self._call_api(prompt)
        
        return {
            'approach': 'Chain of Thought',
            'prompt': prompt,
            'response': response,
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
    
    def tree_of_thought(self, problem):
        """Tree of Thought (ToT) Approach"""
        prompt = f"""Problem: {problem}

I'll explore multiple solution paths and evaluate them:

Path 1 (Direct calculation):
- Steps: ...
- Evaluation: ...

Path 2 (Alternative method):
- Steps: ...
- Evaluation: ...

Path 3 (Verification approach):
- Steps: ...
- Evaluation: ...

Best path analysis:
Final solution:"""

        print(f"\n{'='*60}")
        print("TREE OF THOUGHT (ToT)")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}\n")
        
        response = self._call_api(prompt)
        
        return {
            'approach': 'Tree of Thought',
            'prompt': prompt,
            'response': response,
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
    
    def zero_shot(self, problem):
        """Zero-shot Prompting"""
        prompt = f"""Solve this problem: {problem}"""
        
        print(f"\n{'='*60}")
        print("ZERO-SHOT PROMPTING")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}\n")
        
        response = self._call_api(prompt)
        
        return {
            'approach': 'Zero-shot',
            'prompt': prompt,
            'response': response,
            'prompt_length': len(prompt),
            'response_length': len(response)
        }
    
    def few_shot(self, problem, examples):
        """Few-shot Prompting"""
        examples_text = "\n\n".join([
            f"Example {i+1}:\nProblem: {ex['problem']}\nSolution: {ex['solution']}"
            for i, ex in enumerate(examples)
        ])
        
        prompt = f"""{examples_text}

Now solve this problem:
Problem: {problem}
Solution:"""

        print(f"\n{'='*60}")
        print("FEW-SHOT PROMPTING")
        print(f"{'='*60}")
        print(f"Prompt:\n{prompt}\n")
        
        response = self._call_api(prompt)
        
        return {
            'approach': 'Few-shot',
            'prompt': prompt,
            'response': response,
            'prompt_length': len(prompt),
            'response_length': len(response),
            'num_examples': len(examples)
        }
    
    def _call_api(self, prompt):
        """Call Gemini API or return demo response"""
        if self.demo_mode:
            # Demo responses for illustration
            time.sleep(0.5)  # Simulate API call
            return self._generate_demo_response(prompt)
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"API Error: {str(e)}\n\nDemo response: {self._generate_demo_response(prompt)}"
    
    def _generate_demo_response(self, prompt):
        """Generate demo response for testing without API"""
        if "Interview" in prompt or "questions" in prompt:
            return """1. What do we need to find? We need to find the solution to the given problem.
2. What information do we have? We have the problem statement and context.
3. What steps should we take? Break down the problem, analyze each part, combine solutions.
4. What is the solution? The solution follows from systematic analysis of the problem components."""
        
        elif "step by step" in prompt or "Chain" in prompt:
            return """Step 1: Understanding the problem - I identify the key requirements.
Step 2: Gathering information - I list all given data points.
Step 3: Planning the solution - I outline the methodology.
Step 4: Executing the plan - I work through calculations systematically.
Step 5: Verification - I check the answer makes sense.
Final Answer: The solution is derived through logical progression."""
        
        elif "Path" in prompt or "Tree" in prompt:
            return """Path 1 (Direct): Quick calculation gives us an initial answer.
Evaluation: Fast but may miss edge cases. Score: 7/10

Path 2 (Systematic): Thorough step-by-step breakdown.
Evaluation: More reliable, catches errors. Score: 9/10

Path 3 (Alternative): Using a different method for verification.
Evaluation: Good for cross-checking. Score: 8/10

Best Path: Path 2 (Systematic approach)
Final Solution: Based on the most thorough analysis."""
        
        else:
            return "The solution involves analyzing the problem systematically and applying appropriate methods to reach a conclusion."
    
    def compare_all_approaches(self, problem):
        """Compare all prompt engineering approaches"""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE PROMPT ENGINEERING COMPARISON")
        print(f"{'='*70}")
        
        results = {
            'interview': self.interview_approach(problem),
            'cot': self.chain_of_thought(problem),
            'tot': self.tree_of_thought(problem)
        }
        
        self.results = results
        return results
    
    def compare_shot_types(self, problem, examples):
        """Compare zero-shot vs few-shot"""
        print(f"\n{'='*70}")
        print("ZERO-SHOT VS FEW-SHOT COMPARISON")
        print(f"{'='*70}")
        
        results = {
            'zero_shot': self.zero_shot(problem),
            'few_shot': self.few_shot(problem, examples)
        }
        
        return results
    
    def visualize_comparison(self, results, save_path='prompt_comparison.png'):
        """Visualize prompt engineering results"""
        print(f"\n{'='*60}")
        print("CREATING PROMPT ENGINEERING VISUALIZATIONS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Prompt Lengths
        approaches = [r['approach'] for r in results.values()]
        prompt_lengths = [r['prompt_length'] for r in results.values()]
        response_lengths = [r['response_length'] for r in results.values()]
        
        x = np.arange(len(approaches))
        width = 0.35
        
        bars1 = axes[0].bar(x - width/2, prompt_lengths, width, 
                           label='Prompt Length', color='#FF6B6B', alpha=0.8)
        bars2 = axes[0].bar(x + width/2, response_lengths, width, 
                           label='Response Length', color='#4ECDC4', alpha=0.8)
        
        axes[0].set_xlabel('Approach', fontweight='bold')
        axes[0].set_ylabel('Character Count', fontweight='bold')
        axes[0].set_title('Prompt vs Response Lengths', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(approaches, rotation=15, ha='right')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}',
                           ha='center', va='bottom', fontsize=8)
        
        # Plot 2: Complexity Score (based on prompt structure)
        complexity_scores = []
        for r in results.values():
            score = (
                r['prompt'].count('\n') * 2 +  # Structure
                r['prompt'].count('?') * 3 +    # Questions
                r['prompt'].count(':') * 1.5    # Organization
            )
            complexity_scores.append(score)
        
        bars = axes[1].barh(approaches, complexity_scores, 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[1].set_xlabel('Complexity Score', fontweight='bold')
        axes[1].set_title('Prompt Engineering Complexity', fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, complexity_scores)):
            axes[1].text(score, i, f' {score:.1f}', 
                        va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")
        plt.close()


# ============================================================================
# PART 4: RESULTS EXPORT & REPORTING
# ============================================================================

class ResultsExporter:
    """Export results in multiple formats"""
    
    @staticmethod
    def export_to_json(data, filename='results.json'):
        """Export results to JSON"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Results exported to {filename}")
    
    @staticmethod
    def export_to_csv(data, filename='results.csv'):
        """Export results to CSV"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            for key, value in data.items():
                writer.writerow([key, value])
        print(f"Results exported to {filename}")
    
    @staticmethod
    def generate_report(gmm_metrics, markov_stats, prompt_results, filename='report.txt'):
        """Generate comprehensive text report"""
        with open(filename, 'w') as f:
            f.write("="*70 + "\n")
            f.write("MACHINE LEARNING & NLP ASSIGNMENT - COMPREHENSIVE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. GAUSSIAN MIXTURE MODEL RESULTS\n")
            f.write("-" * 70 + "\n")
            for key, value in gmm_metrics.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n2. MARKOV CHAIN STATISTICS\n")
            f.write("-" * 70 + "\n")
            for key, value in markov_stats.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n3. PROMPT ENGINEERING COMPARISON\n")
            f.write("-" * 70 + "\n")
            for approach, results in prompt_results.items():
                f.write(f"\n{approach.upper()}:\n")
                f.write(f"  Approach: {results['approach']}\n")
                f.write(f"  Prompt Length: {results['prompt_length']}\n")
                f.write(f"  Response Length: {results['response_length']}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write(f"Report generated: {datetime.now()}\n")
            f.write("="*70 + "\n")
        
        print(f"Comprehensive report saved: {filename}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print(" "*15 + "ML & NLP ASSIGNMENT")
    print("="*70)
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # ========================================================================
    # PART 1: GAUSSIAN MIXTURE MODEL
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# PART 1: GAUSSIAN MIXTURE MODEL")
    print("#"*70)
    
    gmm = EnhancedGMM(n_components=3, n_samples=500)
    gmm.generate_data()
    gmm_metrics = gmm.train_model()
    gmm.visualize(save_path='output/gmm_visualization.png')
    
    # Generate new samples
    print(f"\n{'='*60}")
    print("GENERATING NEW SAMPLES FROM LEARNED DISTRIBUTION")
    print(f"{'='*60}")
    new_samples, labels = gmm.generate_new_samples(n_samples=50)
    print(f"✓ Generated {len(new_samples)} new samples")
    
    # ========================================================================
    # PART 2: MARKOV CHAIN TEXT GENERATOR
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# PART 2: MARKOV CHAIN TEXT GENERATOR")
    print("#"*70)
    
    # Sample corpus
    sample_text = """
    Machine learning is a subset of artificial intelligence. Artificial intelligence 
    aims to create intelligent systems. Intelligent systems can learn from data. 
    Data is the new oil in the digital age. The digital age has transformed how we 
    work and live. We live in an era of rapid technological advancement. Technological 
    advancement brings both opportunities and challenges. Challenges help us grow and 
    innovate. Innovation drives progress in science and technology. Science and technology 
    shape our future. The future depends on how we use these tools today. Today we must 
    make responsible choices. Responsible choices lead to better outcomes. Better outcomes 
    improve quality of life. Quality of life is what matters most.
    """
    
    # Test different orders
    for order in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"TESTING MARKOV CHAIN WITH ORDER {order}")
        print(f"{'='*60}")
        
        markov = MarkovChainGenerator(order=order)
        markov.train(sample_text)
        stats = markov.analyze_chain()
        
        print(f"\nGenerated Text (Order {order}):")
        print("-" * 60)
        generated = markov.generate(max_length=40)
        print(generated)
        print("-" * 60)
    
    # ========================================================================
    # PART 3: PROMPT ENGINEERING
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# PART 3: PROMPT ENGINEERING WITH GEMINI API")
    print("#"*70)
    
    # Initialize comparator (use None for demo mode, or provide API key)
    api_key = None  # Replace with: os.environ.get('GEMINI_API_KEY') if you have key
    comparator = PromptEngineeringComparator(api_key=api_key)
    
    if api_key is None:
        print("\n  Running in DEMO MODE (no API key provided)")
        print("   To use real Gemini API, set your API key")
        print("   Example: api_key = 'your-api-key-here'\n")
    
    # Test problem
    problem = """A farmer has 20 animals on his farm, consisting of chickens and cows. 
    If there are 56 legs in total, how many chickens and how many cows does the farmer have?"""
    
    # Compare all approaches
    prompt_results = comparator.compare_all_approaches(problem)
    
    # ========================================================================
    # PART 4: ZERO-SHOT VS FEW-SHOT
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# PART 4: ZERO-SHOT VS FEW-SHOT PROMPTING")
    print("#"*70)
    
    # Few-shot examples
    examples = [
        {
            'problem': 'If 2 apples cost $3, how much do 6 apples cost?',
            'solution': '6 apples = 3 × (2 apples) = 3 × $3 = $9'
        },
        {
            'problem': 'A rectangle has length 8cm and width 5cm. What is its area?',
            'solution': 'Area = length × width = 8 × 5 = 40 cm²'
        }
    ]
    
    shot_results = comparator.compare_shot_types(problem, examples)
    
    # Visualize results
    all_prompt_results = {**prompt_results, **shot_results}
    comparator.visualize_comparison(all_prompt_results, 
                                   save_path='output/prompt_comparison.png')
    
    # ========================================================================
    # PART 5: EXPORT RESULTS
    # ========================================================================
    print("\n\n" + "#"*70)
    print("# PART 5: EXPORTING RESULTS")
    print("#"*70)
    
    exporter = ResultsExporter()
    
    # Export GMM metrics
    exporter.export_to_json(gmm_metrics, 'output/gmm_metrics.json')
    exporter.export_to_csv(gmm_metrics, 'output/gmm_metrics.csv')
    
    # Export Markov stats
    exporter.export_to_json(stats, 'output/markov_stats.json')
    
    # Generate comprehensive report
    exporter.generate_report(gmm_metrics, stats, all_prompt_results, 
                            'output/comprehensive_report.txt')
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n\n" + "="*70)
    print(" "*20 + "ASSIGNMENT COMPLETED!")
    print("="*70)
    print("\n WHAT WAS ACCOMPLISHED:")
    print("  ✓ Gaussian Mixture Model with visualization")
    print("  ✓ Markov Chain Text Generator (multiple orders)")
    print("  ✓ Interview Approach prompting")
    print("  ✓ Chain of Thought (CoT) prompting")
    print("  ✓ Tree of Thought (ToT) prompting")
    print("  ✓ Zero-shot vs Few-shot comparison")
    print("  ✓ Performance metrics and analytics")
    print("  ✓ Visual comparisons and charts")
    print("  ✓ Exported results (JSON, CSV, TXT)")
    
    print("\nOUTPUT FILES GENERATED:")
    print("  - output/gmm_visualization.png")
    print("  - output/prompt_comparison.png")
    print("  - output/gmm_metrics.json")
    print("  - output/gmm_metrics.csv")
    print("  - output/markov_stats.json")
    print("  - output/comprehensive_report.txt")
    
   
    



if __name__ == "__main__":
    main()