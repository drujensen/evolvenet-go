package main

import (
	"math"
	"math/rand"
)

// CMAES implements Covariance Matrix Adaptation Evolution Strategy
type CMAES struct {
	dim    int         // dimension (total weight count)
	mu     int         // parent number
	lambda int         // offspring number
	sigma  float64     // step size
	C      [][]float64 // covariance matrix (dim x dim)
	m      []float64   // mean vector (dim)
	pc     []float64   // evolution path for C
	ps     []float64   // evolution path for sigma
	E      float64     // expectation of ||N(0,I)||
	muEff  float64     // effective mu
	cSigma float64     // learning rate for sigma
	cC     float64     // learning rate for C
	cCum   float64     // cumulation parameter
	dSigma float64     // damping for sigma

	weights []float64

	cMu float64
}

// NewCMAES initializes CMA-ES with given dimension and parameters
func NewCMAES(dim, lambda int) *CMAES {
	mu := lambda / 2
	muEff := float64(mu)

	// Initialize covariance matrix as identity
	C := make([][]float64, dim)
	for i := range C {
		C[i] = make([]float64, dim)
		C[i][i] = 1.0
	}

	// Initialize mean vector (zero)
	m := make([]float64, dim)

	// Evolution paths
	pc := make([]float64, dim)
	ps := make([]float64, dim)

	// CMA-ES parameters
	E := math.Sqrt(float64(dim)) * (1.0 - 1.0/(4.0*float64(dim)) + 1.0/(21.0*float64(dim*dim)))
	cSigma := (muEff + 2.0) / (float64(dim) + muEff + 5.0)
	cC := (4.0 + muEff/float64(dim)) / (float64(dim) + 4.0 + 2.0*muEff/float64(dim))
	cCum := 4.0 / (float64(dim) + 4.0)
	dSigma := 1.0 + cSigma + 2.0*math.Max(0, math.Sqrt((muEff-1.0)/(float64(dim)+1.0))-1.0)

	return &CMAES{
		dim:    dim,
		mu:     mu,
		lambda: lambda,
		sigma:  1.0, // initial step size
		C:      C,
		m:      m,
		pc:     pc,
		ps:     ps,
		E:      E,
		muEff:  muEff,
		cSigma: cSigma,
		cC:     cC,
		cCum:   cCum,
		dSigma: dSigma,
	}
}

// sample generates lambda new candidate solutions
func (cma *CMAES) sample() [][]float64 {
	population := make([][]float64, cma.lambda)

	// Cholesky decomposition of covariance matrix (simplified for diagonal)
	// For now, assume diagonal covariance for simplicity
	B := make([]float64, cma.dim) // square roots of diagonal elements
	for i := 0; i < cma.dim; i++ {
		B[i] = math.Sqrt(cma.C[i][i])
	}

	for i := 0; i < cma.lambda; i++ {
		// Sample from multivariate normal: m + sigma * B * z
		z := make([]float64, cma.dim)
		for j := 0; j < cma.dim; j++ {
			z[j] = rand.NormFloat64() // standard normal
		}

		// For diagonal covariance, just scale by B
		x := make([]float64, cma.dim)
		for j := 0; j < cma.dim; j++ {
			x[j] = cma.m[j] + cma.sigma*B[j]*z[j]
		}

		population[i] = x
	}

	return population
}

// update adapts the distribution based on selected offspring
func (cma *CMAES) update(selected [][]float64) {
	if len(selected) != cma.mu {
		panic("selected population must have mu individuals")
	}

	// Update mean: weighted average of selected
	oldM := make([]float64, cma.dim)
	copy(oldM, cma.m)

	for i := 0; i < cma.dim; i++ {
		sum := 0.0
		for j := 0; j < cma.mu; j++ {
			sum += selected[j][i]
		}
		cma.m[i] = sum / float64(cma.mu)
	}

	// Update evolution paths
	// First, compute m - oldM (evolution direction)
	mDiff := make([]float64, cma.dim)
	for i := 0; i < cma.dim; i++ {
		mDiff[i] = cma.m[i] - oldM[i]
	}

	// Normalize by sigma
	for i := 0; i < cma.dim; i++ {
		mDiff[i] /= cma.sigma
	}

	// Update ps (for sigma control)
	for i := 0; i < cma.dim; i++ {
		cma.ps[i] = (1.0-cma.cSigma)*cma.ps[i] + math.Sqrt(cma.cSigma*(2.0-cma.cSigma)*cma.muEff)*mDiff[i]
	}

	// Update pc (for covariance control)
	for i := 0; i < cma.dim; i++ {
		cma.pc[i] = (1.0-cma.cCum)*cma.pc[i] + math.Sqrt(cma.cCum*(2.0-cma.cCum)*cma.muEff)*mDiff[i]
	}

	// Update step size sigma
	psNorm := 0.0
	for i := 0; i < cma.dim; i++ {
		psNorm += cma.ps[i] * cma.ps[i]
	}
	psNorm = math.Sqrt(psNorm)

	cma.sigma *= math.Exp((psNorm/cma.E - 1.0) * cma.cSigma / cma.dSigma)

	// Update covariance matrix
	// For simplicity, update only diagonal elements (rank-one update)
	for i := 0; i < cma.dim; i++ {
		// Rank-one update: C = (1 - cC) * C + cC * pc * pc^T
		for j := 0; j < cma.dim; j++ {
			cma.C[i][j] *= (1.0 - cma.cC)
		}
		// Add rank-one term (simplified for diagonal)
		cma.C[i][i] += cma.cC * cma.pc[i] * cma.pc[i]
	}
}
