//! Contains the code for batch proving a number of LogUp GKR claims.

use std::sync::Arc;

use ff_ext::ExtensionField;
use itertools::izip;
use multilinear_extensions::{
    mle::{IntoMLE, MultilinearExtension},
    virtual_poly::{ArcMultilinearExtension, VirtualPolynomial},
};
use sumcheck::structs::{IOPProof, IOPProverState};
use transcript::Transcript;

use crate::{Claim, commit::compute_betas_eval, lookup::logup_gkr::circuit::LogUpLayer};

use super::structs::{LogUpProof, LookupInput, TableInput};

/// Function to batch prove a collection of [`LookupInput`]s
/// TODO: add support to batch in claims about the output of this in the final step of the GKR circuit.
pub fn batch_prove_lookups<E: ExtensionField, T: Transcript<E>>(
    lookup_input: &LookupInput<E>,
    transcript: &mut T,
) -> LogUpProof<E> {
    // Work out how many instances we are dealing with
    let circuits = lookup_input.make_circuits();
    let num_instances = circuits.len();

    // Work out the total number of layers and the number of layers per instance.
    let mut total_layers = 0usize;
    let circuit_outputs = circuits
        .iter()
        .map(|c| {
            total_layers = std::cmp::max(total_layers, c.num_vars());
            c.outputs()
        })
        .collect::<Vec<Vec<E>>>();

    // When proving we want to work from the top down so we convert each of the circuits into an iterator over its layers in reverse order.
    // We also skip the first layer after reversing as this is just the output claims.
    let mut layer_iters = circuits
        .iter()
        .map(|c| c.layers().into_iter().rev().skip(1))
        .collect::<Vec<_>>();

    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda
    transcript.append_field_element(&E::BaseField::from(num_instances as u64));
    circuit_outputs
        .iter()
        .for_each(|evals| transcript.append_field_element_exts(evals));

    let batching_challenge = transcript
        .get_and_append_challenge(b"inital_batching")
        .elements;
    let mut alpha = transcript
        .get_and_append_challenge(b"inital_alpha")
        .elements;
    let mut lambda = transcript
        .get_and_append_challenge(b"initial_lambda")
        .elements;

    let (mut current_claim, _) =
        circuit_outputs
            .iter()
            .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                // we have four evals and we batch them as alpha * (batching_challenge * (e[1] - e[0]) + e[0] + lambda * (batching_challenge * (e[3] - e[2]) + e[2]) )
                (
                    acc + alpha_comb
                        * (batching_challenge * (e[1] - e[0])
                            + e[0]
                            + lambda * (batching_challenge * (e[3] - e[2]) + e[2])),
                    alpha_comb * alpha,
                )
            });

    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    let mut sumcheck_proofs: Vec<IOPProof<E>> = vec![];

    let mut round_evaluations: Vec<Vec<E>> = vec![];

    for current_layer_vars in 1..=total_layers {
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Compute the eq_evals if we aren't in the first round
        let eq_poly: ArcMultilinearExtension<E> =
            Arc::new(compute_betas_eval(&sumcheck_point).into_mle());

        // Then we progress through the `current_claims` adding to the virtual polynomial if we have something to prove
        let mut vp = VirtualPolynomial::<E>::new(current_layer_vars);

        let alpha_powers = std::iter::successors(Some(E::ONE), |prev| Some(*prev * alpha))
            .take(layer_iters.len())
            .collect::<Vec<E>>();

        izip!(layer_iters.iter_mut(), alpha_powers).for_each(|(iter, a)| {
            let layer = iter.next().unwrap();
            let mles = layer.get_mles();
            if let LogUpLayer::Generic { .. } = layer {
                vp.add_mle_list(vec![eq_poly.clone(), mles[0].clone(), mles[3].clone()], a);
                vp.add_mle_list(vec![eq_poly.clone(), mles[1].clone(), mles[2].clone()], a);
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[2].clone(), mles[3].clone()],
                    a * lambda,
                );
            } else {
                vp.add_mle_list(vec![eq_poly.clone(), mles[1].clone()], -a);
                vp.add_mle_list(vec![eq_poly.clone(), mles[0].clone()], -a);
                vp.add_mle_list(
                    vec![eq_poly.clone(), mles[0].clone(), mles[1].clone()],
                    a * lambda,
                );
            }
        });

        // Run the sumcheck for this round
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);

        // Update the sumcheck proof
        sumcheck_point = proof.point.clone();

        // The first one is always the eq_poly eval
        let evals = &state.get_mle_final_evaluations()[1..];

        // Squeeze the challenges to combine everything into a single sumcheck
        let batching_challenge = transcript
            .get_and_append_challenge(b"logup_batching")
            .elements;
        alpha = transcript.get_and_append_challenge(b"logup_alpha").elements;
        lambda = transcript
            .get_and_append_challenge(b"logup_lambda")
            .elements;
        // Append the batching challenge to the proof point
        sumcheck_point.push(batching_challenge);
        // Append the sumcheck proof to the list of proofs
        sumcheck_proofs.push(proof);

        current_claim = if current_layer_vars != total_layers {
            evals
                .chunks(4)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                    (
                        acc + alpha_comb
                            * (batching_challenge * (e[2] - e[0])
                                + e[0]
                                + lambda * (batching_challenge * (e[1] - e[3]) + e[3])),
                        alpha_comb * alpha,
                    )
                })
                .0
        } else {
            evals
                .chunks(2)
                .fold((E::ZERO, E::ONE), |(acc, alpha_comb), e| {
                    (
                        acc + alpha_comb * (batching_challenge * (e[0] - e[1]) + e[1]),
                        alpha_comb * alpha,
                    )
                })
                .0
        };

        round_evaluations.push(evals.to_vec());
    }

    let output_claims = lookup_input
        .base_mles()
        .iter()
        .map(|mle| Claim::<E> {
            point: sumcheck_point.clone(),
            eval: mle.evaluate(&sumcheck_point),
        })
        .collect::<Vec<Claim<E>>>();

    LogUpProof::<E> {
        sumcheck_proofs,
        round_evaluations,
        output_claims,
        circuit_outputs,
    }
}

/// Function to batch prove a collection of [`LogUpCircuit`]s
pub fn prove_table<E: ExtensionField, T: Transcript<E>>(
    table_input: &TableInput<E>,
    transcript: &mut T,
) -> LogUpProof<E> {
    // Create the circuit
    let circuit = table_input.make_circuit();

    // Work out the total number of layers and the number of layers per instance.
    let total_layers = circuit.num_vars();
    let circuit_outputs = circuit.outputs();

    // When proving we want to work from the top down so we convert each of the circuits into an iterator over its layers in reverse order.
    // We also skip the first layer after reversing as this is just the output claims.
    let mut layer_iter = circuit.layers().into_iter().rev().skip(1);

    // Append the number of instances along with their output evals to the transcript and then squeeze our first alpha and lambda

    transcript.append_field_element_exts(&circuit_outputs);

    let batching_challenge = transcript
        .get_and_append_challenge(b"inital_batching")
        .elements;

    let mut lambda = transcript
        .get_and_append_challenge(b"initial_lambda")
        .elements;

    let mut current_claim = batching_challenge * (circuit_outputs[1] - circuit_outputs[0])
        + circuit_outputs[0]
        + lambda
            * (batching_challenge * (circuit_outputs[3] - circuit_outputs[2]) + circuit_outputs[2]);

    // The initial sumcheck point is just the batching challenge
    let mut sumcheck_point: Vec<E> = vec![batching_challenge];

    let mut sumcheck_proofs: Vec<IOPProof<E>> = vec![];

    let mut round_evaluations: Vec<Vec<E>> = vec![];

    for current_layer_vars in 2..total_layers {
        // Append the current claim to the transcript
        transcript.append_field_element_ext(&current_claim);

        // Compute the eq_evals if we aren't in the first round
        let eq_poly: ArcMultilinearExtension<E> =
            Arc::new(compute_betas_eval(&sumcheck_point).into_mle());

        // Then we progress through the `current_claims` adding to the virtual polynomial if we have something to prove
        let mut vp = VirtualPolynomial::<E>::new(current_layer_vars);

        let layer = layer_iter.next().unwrap();
        let mles = layer.get_mles();

        vp.add_mle_list(
            vec![eq_poly.clone(), mles[0].clone(), mles[3].clone()],
            E::ONE,
        );
        vp.add_mle_list(
            vec![eq_poly.clone(), mles[1].clone(), mles[2].clone()],
            E::ONE,
        );
        vp.add_mle_list(
            vec![eq_poly.clone(), mles[2].clone(), mles[3].clone()],
            lambda,
        );

        // Run the sumcheck for this round
        #[allow(deprecated)]
        let (proof, state) = IOPProverState::<E>::prove_parallel(vp, transcript);

        // Update the sumcheck proof
        sumcheck_point = proof.point.clone();

        // The first one is always the eq_poly eval
        let evals = &state.get_mle_final_evaluations()[1..];

        // Squeeze the challenges to combine everything into a single sumcheck
        let batching_challenge = transcript
            .get_and_append_challenge(b"logup_batching")
            .elements;

        lambda = transcript
            .get_and_append_challenge(b"logup_lambda")
            .elements;
        // Append the batching challenge to the proof point
        sumcheck_point.push(batching_challenge);
        // Append the sumcheck proof to the list of proofs
        sumcheck_proofs.push(proof);

        current_claim = batching_challenge * (evals[2] - evals[0])
            + evals[0]
            + lambda * (batching_challenge * (evals[1] - evals[2]) + evals[2]);

        round_evaluations.push(evals.to_vec());
    }

    let output_claims = table_input
        .base_mles()
        .iter()
        .map(|mle| Claim::<E> {
            point: sumcheck_point.clone(),
            eval: mle.evaluate(&sumcheck_point),
        })
        .collect::<Vec<Claim<E>>>();

    LogUpProof::<E> {
        sumcheck_proofs,
        round_evaluations,
        output_claims,
        circuit_outputs: vec![circuit_outputs],
    }
}
