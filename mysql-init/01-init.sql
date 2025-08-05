DROP TABLE IF EXISTS assignments;

CREATE TABLE assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id VARCHAR(255) NOT NULL,
    course_title VARCHAR(255) NOT NULL,
    instructor_name VARCHAR(255) NOT NULL,
    content MEDIUMTEXT NOT NULL,
    topic VARCHAR(100) NOT NULL
);

INSERT INTO assignments (course_id, course_title, instructor_name, content, topic) VALUES
('PEF301', 'Power Electronics Lab', 'Dr. G. Rao', '1. Design the switching model of the boost converter. Refer: Lab Lecture:01 [5]\n2. Design a buck converter using different combinations of capacitors to output voltage ripple. Refer: Lab Lecture:02 [5]\n3. Electric Vehicle Demonstration [5] Refer to Lab Lecture:02\n4. Sizing and scaling of three vehicles using different drive cycles with Dorleco Sim EV. Refer to Lab Lecture:02 [5]', 'Boost and Buck Converter Design'),

('CSD345', 'Mobile Application Development', 'Dr. D. Ramesh', 'Create a Flutter app using the sqflite plugin that:\n- Supports adding, updating, and deleting data about Persons.\n- Works on both Android and iOS.\n- Includes persistent local database storage.\nEnsure proper UI and test CRUD operations. Review SQLite basics before implementing.', 'SQLite in Flutter'),

('DSP203', 'Digital Signal Processing', 'Dr. A. Kulkarni', 'Let x(n) = {1,2,3,4,3,2,1}, h(n) = {-1,0,1}\n(a) Compute convolution y(n) = x(n)*h(n) using matrix-vector method\n(b) Implement MATLAB function convtoep(x,h) using Toeplitz matrix\n(c) Verify your function using above sequences.', 'Convolution and Filtering'),

('DSP505', 'Advanced DSP Systems', 'Dr. Shankar Iyer', 'Design a lowpass filter using Pade and Prony methods:\n(a) Cutoff freq c=2, p+q+1=20; vary p=0..20 and analyze\n(b) Repeat for c=16 (narrowband)\n(c) Apply Prony method to minimize error\n(d) Compare with elliptic filter using ellip.m\nWrite analysis comparing designs.', 'Filter Design with Pade and Prony'),

('CSF363', 'Compiler Construction', 'Dr. Meena R.', '1. Construct a shift-reduce parser for a given CFG.\n2. Explain difference between top-down and bottom-up parsing.\n3. Trace input string parsing using created parser.', 'Parsing Techniques'),

('CSF363', 'Compiler Construction', 'Dr. Meena R.', 'Design regular expressions for:\n1. Binary strings ending in 01\n2. All strings with even number of 1s\n3. Convert NFA to DFA and minimize.', 'Regular Expressions'),

('CSF265', 'Machine Learning', 'Dr. Anirudh Gupta', '1. Implement Logistic Regression from scratch\n2. Plot decision boundary\n3. Explain overfitting and regularization using L2 penalty\n4. Classify Iris dataset using your model.', 'Logistic Regression'),

('CSF265', 'Machine Learning', 'Dr. Anirudh Gupta', '1. Design a 3-layer neural network for MNIST classification\n2. Implement ReLU and Sigmoid activations\n3. Visualize hidden layer outputs\n4. Perform training and report accuracy.', 'Neural Networks'),

('MEF112', 'Thermal Systems', 'Dr. S. Venkat', '1. Derive first law for closed system.\n2. Explain entropy change in reversible process\n3. Calculate Carnot efficiency for given heat sources.', 'Thermodynamics'),

('MEF112', 'Thermal Systems', 'Dr. S. Venkat', '1. Define and derive Fourier Law\n2. Design finned surface for heat dissipation\n3. Analyze steady-state temperature profile of a rod.', 'Heat Transfer'),

('ECF211', 'Basic Electrical Circuits', 'Dr. A. Rao', '1. Use nodal analysis to find circuit voltages\n2. Apply Thevenin theorem for equivalent circuit\n3. Use mesh analysis to find loop currents', 'Circuit Analysis'),

('ECF211', 'Digital Systems Design', 'Dr. A. Rao', '1. Design 4-bit binary adder using full adders\n2. Convert JK to D Flip-Flop\n3. Implement 3-to-8 decoder using gates.', 'Digital Logic'),

('BIO113', 'Genetics', 'Dr. R. Malhotra', '1. Draw Punnett square for dihybrid cross\n2. Explain Mendel’s 3 laws with examples\n3. Identify gene linkage and recombination scenarios.', 'Mendelian Genetics'),

('BIO113', 'Cell Biology', 'Dr. R. Malhotra', '1. Diagram mitosis and meiosis with phases\n2. Compare their roles in organism development\n3. Explain checkpoints and regulation of cell cycle.', 'Cell Cycle'),

('CHEM112', 'Inorganic Chemistry', 'Dr. A. Bansal', '1. Draw Lewis structure of SO2\n2. Compare ionic and covalent bonds\n3. Use VSEPR to predict shapes of SF6, NH3, CH4', 'Chemical Bonding'),

('CHEM112', 'Thermodynamics in Chemistry', 'Dr. A. Bansal', '1. Apply Hess’s Law to multistep reactions\n2. Calculate ΔH using calorimetry\n3. Explain standard enthalpy of formation.', 'Thermochemistry'),

('MATH112', 'Calculus I', 'Dr. J. Narayanan', '1. Use chain rule to differentiate composite functions\n2. Find volume of solid using integration\n3. Evaluate limits using L’Hôpital’s Rule.', 'Differentiation and Integration'),

('MATH112', 'Linear Algebra', 'Dr. J. Narayanan', '1. Perform Gaussian Elimination\n2. Compute eigenvalues and eigenvectors\n3. Use Rank-Nullity theorem on a matrix.', 'Matrix Theory'),

('ECONC313', 'Economics and Strategy', 'Dr. Nidhi G.', '1. Model the Prisoner’s Dilemma\n2. Identify Nash Equilibrium\n3. Compare cooperative vs non-cooperative games.', 'Game Theory'),

('ECONC313', 'Public Economics', 'Dr. Nidhi G.', '1. Define public goods and their characteristics\n2. Explain free-rider problem\n3. Use diagram to show optimal provision of public goods.', 'Public Goods'),

('BIO212', 'Biochemistry', 'Dr. Leela Sharma', '1. Plot Michaelis-Menten curve\n2. Calculate Km and Vmax from Lineweaver-Burk plot\n3. Describe enzyme-substrate complex.', 'Enzymology'),

('BIO212', 'Immunology', 'Dr. Leela Sharma', '1. Diagram antigen-antibody binding\n2. List functions of IgA, IgG, IgM\n3. Compare innate vs adaptive immune responses.', 'Immune System'),

('CSF213', 'Operating Systems', 'Dr. S. Nair', '1. Implement FCFS and Round Robin scheduling\n2. Compare performance using Gantt chart\n3. Explain context switching and its overhead.', 'Process Scheduling'),

('CSF213', 'Operating Systems', 'Dr. S. Nair', '1. Explain paging, segmentation with diagrams\n2. Analyze TLB hit/miss rate\n3. Compare memory allocation strategies.', 'Memory Management'),

('HSSF125', 'Ethics and Society', 'Dr. L. Kapoor', '1. Write a critique of autonomous weapon ethics\n2. Discuss algorithmic bias in facial recognition\n3. Suggest an ethical AI framework.', 'Ethics in AI'),

('HSSF125', 'Critical Thinking', 'Dr. L. Kapoor', '1. Identify logical fallacies in argument\n2. Rewrite flawed argument correctly\n3. Write a short editorial using valid reasoning.', 'Argument Analysis'),

('POE111', 'Philosophy of Science', 'Dr. A. Das', '1. Define hypothesis, theory, and law\n2. Apply scientific method to real-world case\n3. Compare deductive vs inductive reasoning.', 'Scientific Reasoning'),

('POE111', 'Logic and Argumentation', 'Dr. A. Das', '1. Identify fallacies: strawman, ad hominem, slippery slope\n2. Write original arguments using syllogism\n3. Assess argument validity.', 'Logical Fallacies'),

('EEE221', 'Control Systems', 'Dr. R. Thomas', '1. Derive transfer function for DC motor\n2. Sketch root locus for a second-order system\n3. Analyze closed-loop stability.', 'System Modeling'),

('EEE221', 'Signals and Systems', 'Dr. R. Thomas', '1. Compute convolution of signals\n2. Derive Fourier Transform of rect(t)\n3. Determine if given system is LTI and causal.', 'Signal Analysis');
